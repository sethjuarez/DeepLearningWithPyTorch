import re
import json
import torch
from IPython.display import Javascript

def init():
    j = Javascript("""
require.config({
    paths: {
        
        d3: 'https://d3js.org/d3.v5.min'
    }
});

require.undef('tree');

define('tree', ['d3'], function(d3) {
    const draw = function(element, data) {
        var root = d3.hierarchy(data);
        
        var width = 600;
        var height = root.height * 40;
        var radius = 10;
        
        var treeLayout = d3.tree()
            .size([width, height]);
        
        treeLayout(root);

        var svg = d3.select(element).append('svg')
            .attr('width', width+50)
            .attr('height', height+50);

        var transform = svg.append('g')
            .attr('transform', 'translate(20, 20)');
        
        // Links
        transform.append('g')
            .selectAll('line')
            .data(root.links())
            .enter()
            .append('line')
            .attr('stroke', '#c0c0c0')
            .attr('x1', function(d) {return d.source.x;})
            .attr('y1', function(d) {return d.source.y;})
            .attr('x2', function(d) {return d.target.x;})
            .attr('y2', function(d) {return d.target.y;});
        
        // Nodes
        transform.append('g')
            .selectAll('circle')
            .data(root.descendants())
            .enter()
            .append('circle')
            .attr('stroke', function(d) { return d.data.leaf === 'true' ? '#808080' : '#000000'; })
            .attr('fill', function(d) { 
                if(d.data.name == 'Const')
                    return '#B37D4E';
                else if(d.data.name == 'Var')
                    return '#CD5360';
                else
                    return '#286DA8';
             })
            .attr('cx', function(d) {return d.x;})
            .attr('cy', function(d) {return d.y;})
            .attr('r', radius);

        // labels
        transform.append('g')
            .selectAll('text')
            .data(root.descendants())
            .enter()
            .append('text')
            .text(function(d){return d.data.name;})
            .attr('x', function(d) {return d.x+radius+3;})
            .attr('y', function(d) {return d.y+5;});
    };
    return draw;
});
""")
    return j

def build_graph(g):
    tree = {'leaf': 'true'}
    if g == None:
        tree['name'] = 'Const'
    elif hasattr(g, 'variable'):
        tree['name'] = 'Var'
    else:
        name = g.name()
        m = name[:re.search(r'^([^A-Z]*[A-Z]){2}', name).span()[1]-1]
        tree['name'] = m
        
    if g != None and g.next_functions != None:
        children = [build_graph(subg[0]) for subg in g.next_functions]
        if len(children) > 0:
            tree['children'] = children
            tree['leaf'] = 'false'
    
    return tree

def draw_graph(graph):
    return Javascript("""
(function(element){
    require(['tree'], function(tree) {
        tree(element.get(0), %s)
    });
})(element);
""" % json.dumps(graph))

def draw(g):
    graph = build_graph(g.grad_fn)
    return draw_graph(graph)