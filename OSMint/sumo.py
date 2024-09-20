import os
from math import sin, cos, pi as PI
from pathlib import Path
from typing import List, Dict, Union
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from string import ascii_uppercase as ALPHABET

def gen_net(edges_description: List[Dict[str, any]],
            working_dir: Path = Path(''),
            output: Union[Path, str] = 'output.net.xml',
            open_netedit: bool = False) -> Path:
    """

    :param edges_description: description of the net, format:
            [
                {
                    'num_incoming':
                    'num_outgoing':
                    'length': (in m)
                    'speed': (in m/s)
                    'slope': (in radians, positive if incoming vehicles go uphill)
                    'connections': {
                        lane: [(edge, lane), (edge, lane), ...],
                        1: [(2, 0)] # ex : lane 1 will be connected only to the outgoing lane 0 of edge 2
                        ...
                    }
                },
                ...
            ]
            The index is the index in the given array, the lanes are created clockwise (starting north, but it doesn't
            make a difference).
    :param working_dir: where to store temp files
    :param output: either name of the output file (a str) or the full path (a Path). Filename in .net.xml
    :param open_netedit: opens netedit if the net is successfully generated, useful for testing.
    :return: the output path
    """

    nodes = Element('nodes')

    # traffic light in the center
    nodes.append(Element('node', {
        'id': 'TL',
        'x': '0',
        'y': '0',
        'z': '0',
        'type': 'traffic_light'
    }))

    # nodes at the boundaries
    # starts by the West, goes in trigonometric order
    for i, edge in enumerate(edges_description):
        nodes.append(Element('node', {
            'id': ALPHABET[i],
            'x': str(edge['length'] * sin(2 * PI * i / len(edges_description))),
            'y': str(edge['length'] * cos(- 2 * PI * i / len(edges_description))),
            'z': str(- edge['length'] * sin(edge['slope'])),
            'type': 'priority',
        }))

    edges = Element('edges')
    for i, edge in enumerate(edges_description):
        node_id = ALPHABET[i]
        incoming_edge = Element('edge', {
            'id': f'{node_id}2TL',
            'from': node_id,
            'to': 'TL',
            'speed': str(edge['speed']),
            'numLanes': str(edge['num_incoming'])
        })

        outgoing_edge = Element('edge', {
            'id': f'TL2{node_id}',
            'from': 'TL',
            'to': node_id,
            'speed': str(edge['speed']),
            'numLanes': str(edge['num_outgoing']),
        })

        edges.append(incoming_edge)
        edges.append(outgoing_edge)

    connections = Element('connections')
    for edge_index, edge in enumerate(edges_description):
        for incoming_lane, outgoing_connections in edge['connections'].items():
            for outgoing_edge, outgoing_lane in outgoing_connections:
                connections.append(Element('connection', {
                    'from': f'{ALPHABET[edge_index]}2TL',
                    'to': f'TL2{ALPHABET[outgoing_edge]}',
                    'fromLane': str(incoming_lane),
                    'toLane': str(outgoing_lane),
                }))

    for element, filename in [
        (nodes, 'nodes.nod.xml'),
        (edges, 'edges.edg.xml'),
        (connections, 'connections.con.xml')
    ]:
        ElementTree.ElementTree(element).write(working_dir / filename)

    if isinstance(output, str):
        output = working_dir / output

    result = os.system(f'netconvert -v '
                    f'--node-files={working_dir / "nodes.nod.xml"} '
                    f'--edge-files={working_dir / "edges.edg.xml"} '
                    f'--connection-files={working_dir / "connections.con.xml"} '
                    f'--output-file={output}')
    if result==0:
        print('Success')
        if open_netedit:
            os.system(f'netedit {output}')
    else:
        print('Failure, try again :(')

    return output