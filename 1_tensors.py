#%%
import torch
from utils.draw import draw_squares
from utils.square import SquareDataset

# make some pictures!
squares = SquareDataset(120)
for i in range(5):
   print(squares[i])

#%%
# let's look at them
draw_squares(squares)

#%%
# what happens when we multiply?
real_w = torch.tensor([[1,1,1,0,0,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0],
                        [0,0,0,0,0,0,1,1,1]], 
                        dtype=torch.float)

d = ['top', 'middle', 'bottom']
picture, answer = squares[10]
a = d[torch.argmax(answer).item()]
h = picture.view(1, 9).mm(real_w.transpose(0, 1))
b = d[torch.argmax(h).item()]
print('{}\ntruth: {}\nguess: {}'.format(picture, a, b))

