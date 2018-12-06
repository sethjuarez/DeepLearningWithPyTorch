import json
import azureml
from azureml.core.model import Model
from azureml.core import Workspace, Run
from azureml.core.image import ContainerImage, Image
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import Webservice, AciWebservice

def load_workspace():
    # use this code to set up config file
    #subscription_id ='<SUB ID>'
    #resource_group ='<RESOURCE>'
    #workspace_name = '<WORKSPACE>'

    #try:
    #   ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    #   ws.write_config()
    #   print('Workspace configuration succeeded. You are all set!')
    #   return ws
    #except:
    #   print('Workspace not found. TOO MANY ISSUES!!!')

    ws = Workspace.from_config()
    return ws

def main():
    # get workspace
    ws = load_workspace()
    model = Model.register(ws, model_name='pytorch_mnist', model_path='model.pth')
    
    # create dep file
    myenv = CondaDependencies()
    myenv.add_pip_package('numpy')
    myenv.add_pip_package('torch')
    with open('pytorchmnist.yml','w') as f:
        print('Writing out {}'.format('pytorchmnist.yml'))
        f.write(myenv.serialize_to_string())
        print('Done!')

    # create image
    image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                    runtime="python", 
                                    conda_file="pytorchmnist.yml",
                                    dependencies=['./models.py'])

    image = Image.create(ws, 'pytorchmnist', [model], image_config)
    image.wait_for_creation(show_output=True)

    # create service
    aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                                memory_gb=1, 
                                                description='simple MNIST digit detection')
    service = Webservice.deploy_from_image(workspace=ws, 
                                        image=image, 
                                        name='pytorchmnist-svc', 
                                        deployment_config=aciconfig)
    service.wait_for_deployment(show_output=True)

def debug_deploy():
    # get workspace
    ws = load_workspace()
    # get service
    service = ws.webservices['pytorchmnist-svc']
    # write log
    with open('deploy.log','w') as f:
        f.write(service.get_logs())


if __name__ == '__main__':
    # check core SDK version number
    print("Using Azure ML SDK Version: ", azureml.core.VERSION)
    main()
