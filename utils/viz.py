import re
import json
import torch
import altair as alt
from IPython.display import Javascript, HTML


def init():
    return HTML('<script src="https://cdn.jsdelivr.net/npm/vega@5"></script><span>torchviz initialized!</span>')

def get_dict(g, depth):
    d = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": 650,
        "height": depth*30,
        "padding": 5,    
        "signals": [
            {
                "name": "labels", "value": True,
                "bind": {"input": "checkbox"}
            },
            { 
                "name": "method", "value": "tidy",
                "bind": {"input": "select", "options": ["tidy", "cluster"]} 
            },
            { 
                "name": "separation", "value": False, 
                "bind": {"input": "checkbox"} 
            }
        ],
    
        "data": [
            {
                "name": "tree",
                "url": "",
                "values":[ ],
                "transform": [
                    {
                        "type": "stratify",
                        "key": "id",
                        "parentKey": "parentId"
                    },
                    {
                        "type": "tree",
                        "method": {"signal": "method"},
                        "separation": {"signal": "separation"},
                        "size": [{"signal": "width"}, {"signal": "height"}]
                    }]
            },
            {
                "name": "links",
                "source": "tree",
                "url": "",
                "transform": [
                    { "type": "treelinks" },
                    { "type": "linkpath" }
                ]
            }
        ],
        "marks": [
        {
            "type": "path",
            "from": {"data": "links"},
                "encode": {
                "enter": {
                    "stroke": {"value": "#ccc"}
                },
                "update": {
                    "path": {"field": "path"}
                }
            }
        },
        {
            "type": "symbol",
            "from": {"data": "tree"},
            "encode": {
                "enter": {
                    "text": {"field": "id"},
                    "fontSize": {"value": 10},
                    "baseline": {"value": "middle"},
                    "fill": {"field": "color"},
                    "stroke": {"value": "#808080"},
                    "size": {"value": 600 }
                },
                "update": {
                    "x": {"field": "x"},
                    "y": {"field": "y"}
                }
            }
        },
        {
            "type": "text",
            "from": {"data": "tree"},
            "encode": {
            "enter": {
                "text": {"field": "name"},
                "fontSize": {"value": 15},
                "baseline": {"value": "middle"}
            },
            "update": {
                "x": {"field": "x"},
                "y": {"field": "y"},
                "dx": {"signal": "15"},
                "dy": {"signal": "5"},
                "opacity": {"signal": "labels ? 1 : 0"}
            }
            }
        }]
    }
    d['data'][0]['values'] = g
    return d

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

def draw_graph(graph):
    return Javascript("""
(function(element){
    var view = new vega.View(vega.parse(%s), {
        rendered: 'canvas',
        container: element,
        hover: true
    });
    return view.runAsync();
})(element);
""" % json.dumps(graph))

def draw(g):
    graph, depth = build_graph(g.grad_fn, elements=[])
    j = get_dict(graph, depth)
    return draw_graph(j)