import re
import json
import torch
import graphviz
from IPython.display import Javascript, HTML

def build_graph(g, elements=[], parentId=-1, depth=0):
    elm = { 'id': len(elements), 'parentId': None if parentId==-1 else parentId}
    if g == None:
        elm['name'] = 'Const'
        elm['color'] = '#B37D4E'
    elif hasattr(g, 'variable'):
        elm['name'] = 'Var'
        elm['color'] = '#CD5360'
    else:
        name = g.name()
        m = name[:re.search(r'^([^A-Z]*[A-Z]){2}', name).span()[1]-1]
        elm['name'] = m
        elm['color'] = '#286DA8'

    elements.append(elm)
        
    if g != None and g.next_functions != None:
        depth = depth+1
        for subg in g.next_functions:
            elements, depth = build_graph(subg[0], elements, elm['id'], depth)
    
    return elements, depth


def draw(g):
    graph, depth = build_graph(g.grad_fn, elements=[])
    g = graphviz.Digraph('g')
    g.attr('graph', pack='true')
    for item in graph:
        shape = 'rect'
        if item['name'] == 'Const':
            shape='rect'
        if item['name'] == 'Var':
            shape='rect'
        g.attr('node', style='filled', fillcolor=item['color'], 
            color='#303030', fontcolor='white', fontname='Segoe UI', 
            fontsize='10', fixedsize='false', shape=shape, height='0.2',
            width='0.2')
        g.node(str(item['id']), label=item['name'])
        
    for item in graph:
        if item['parentId'] != None:
            g.attr('edge', arrowsize='0.5', color='#303030')
            g.edge(str(item['parentId']), str(item['id']))

    return g
   