"""
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 5A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
******************************************************************************************
"""

"""

* Team Id : 1267

* Author List : Adithya G Jayanth, Ananya Sharma, Ankith L, Ilaa S Chenjeri

* Filename: path.py

* Theme: Geo Guide

* Functions: 
[
    dijkstra,
    path_find,
    gendirect
]

* Global Variables: 
[
    graph,
    directions
]

"""


class path_gen:
    graph = {
        0: {1: 1.5},
        1: {4: 2.5, 2: 2, 0: 1.5},
        2: {3: 3.5, 1: 2},
        3: {2: 3.5, 5: 2.8, 7: 8},
        4: {1: 2.5, 5: 5.5, 8: 3},
        5: {4: 5.5, 10: 2.5, 6: 3, 3: 2.8},
        6: {7: 2.5, 5: 3},
        7: {6: 2.5, 12: 2.5, 3: 8},
        8: {13: 2, 9: 1.8, 4: 3},
        9: {10: 3.5, 8: 1.8},
        10: {9: 3.5, 14: 2, 11: 3, 5: 2.5},
        11: {12: 2.5, 10: 3},
        12: {11: 2.5, 15: 2, 7: 2.5},
        13: {8: 2, 16: 4, 14: 5.5},
        14: {13: 5.5, 15: 5.5, 10: 2},
        15: {14: 5.5, 12: 2, 16: 10.5},
        16: {13: 4, 17: 4},
        17: {15: 2},
    }

    directions = {
        "0 1 4": "l",
        "0 1 2": "r",
        "1 2 3": "s",
        "1 4 8": "s",
        "1 4 5": "r",
        "2 1 0": "l",
        "2 1 4": "r",
        "2 3 5": "l",
        "2 3 7": "s",
        "3 7 6": "l",
        "3 7 12": "s",
        "3 2 1": "s",
        "3 5 6": "r",
        "3 5 4": "l",
        "3 5 10": "s",
        "4 1 0": "s",
        "4 1 2": "l",
        "4 5 6": "s",
        "4 5 10": "l",
        "4 8 9": "r",
        "4 8 13": "s",
        "4 5 3": "r",
        "5 3 2": "r",
        "5 3 7": "l",
        "5 6 7": "s",
        "5 4 1": "l",
        "5 4 8": "r",
        "5 10 11": "r",
        "5 10 9": "l",
        "5 10 14": "s",
        "6 5 3": "l",
        "6 5 10": "r",
        "6 5 4": "s",
        "6 7 3": "r",
        "6 7 12": "l",
        "7 3 2": "s",
        "7 6 5": "s",
        "7 12 15": "s",
        "7 12 11": "l",
        "7 3 5": "r",
        "8 4 1": "s",
        "8 9 10": "s",
        "8 13 16": "s",
        "8 13 14": "r",
        "8 4 5": "l",
        "9 8 13": "r",
        "9 8 4": "l",
        "9 10 14": "l",
        "9 10 5": "r",
        "9 10 11": "s",
        "10 11 12": "s",
        "10 9 8": "s",
        "10 14 15": "r",
        "10 14 13": "l",
        "10 5 6": "l",
        "10 5 4": "r",
        "10 5 3": "s",
        "11 10 9": "s",
        "11 10 14": "r",
        "11 10 5": "l",
        "11 12 7": "r",
        "11 12 15": "l",
        "12 11 10": "s",
        "12 15 14": "l",
        "12 7 6": "r",
        "12 7 3": "s",
        "12 15 16": "s",
        "13 8 4": "s",
        "13 8 9": "l",
        "13 14 15": "s",
        "13 14 10": "r",
        "13 16 15": "s",
        "14 13 16": "r",
        "14 15 16": "l",
        "14 10 9": "r",
        "14 10 11": "l",
        "14 15 12": "r",
        "14 13 8": "l",
        "14 10 5": "s",
        "15 14 13": "s",
        "15 12 7": "s",
        "15 16 13": "s",
        "15 14 10": "l",
        "15 12 11": "r",
        "16 13 8": "s",
        "16 13 14": "l",
        "16 15 12": "s",
        "16 15 14": "r",
        "13 16 17": "s",
        "16 17 15": "s",
        "17 16 13": "s",
        "15 17 16": "s",
        "17 15 12": "s",
        "12 15 17": "s",
    }

    """
        * Function Name: dijkstra
        * Input:
            * `self`: Reference to the object of the class where the function is defined. 
            It is assumed that the class has a member named `graph` which represents 
            the weighted graph as a dictionary of dictionaries. 
            * `start`: The starting node for the shortest path search.
            * `end`: The destination node for the shortest path search.
        * Output:
            * A tuple containing two elements:
                * The shortest path from `start` to `end` as a list of nodes.
                * The distance of the shortest path.
        * Logic:
            * Implements Dijkstra's algorithm to find the shortest path between two nodes 
            in a weighted graph.
            * Uses a greedy approach to iteratively select the unvisited node with the 
            minimum distance from the starting node.
            * Updates distances and previous nodes for neighboring nodes based on the 
            current node.
            * Reconstructs the shortest path by backtracking from the destination node.
        * Example Call:
        shortest_path, distance = dijkstra(graph, 'A', 'D')
    """

    def dijkstra(self, start, end):
        distances = {node: float("inf") for node in self.graph}
        distances[start] = 0
        previous = {node: None for node in self.graph}
        visited = set()

        while len(visited) != len(self.graph):
            min_node = None
            min_distance = float("inf")
            for node in self.graph:
                if node not in visited and distances[node] < min_distance:
                    min_node = node
                    min_distance = distances[node]

            visited.add(min_node)

            for neighbor, weight in self.graph[min_node].items():
                distance = min_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = min_node

        path = []
        current = end
        while current is not None:
            path.insert(0, current)
            current = previous[current]

        return path, distances[end]

    """
        * Function Name: path_find
        * Input:
            * `self`: Reference to the object of the class where the function is defined. It is 
            assumed that the class has a `dijkstra` method and a member named `graph`.
            * `path_list`: A list of node names representing the desired path to be found.
        * Output:
            * A list of node names representing the shortest path between the first and last nodes 
            in `path_list`, incorporating the `graph` information.
        * Logic:
            1. Initialize an empty path list and a previous node variable.
            2. Create a dictionary mapping node names to their corresponding distances from the 
            starting node (assumed to be 0).
            3. Iterate through each node name in `path_list`:
                * Use the `dijkstra` method to find the shortest path from the previous node to 
                the current node.
                * Remove the starting node from the returned path.
                * Append the remaining nodes in the path to the overall path list.
                * Update the previous node to the current node's distance from the starting node 
                (using the dictionary).
            4. Use the `dijkstra` method again to find the shortest path from the last node in the 
            path to the starting node (assumed to be 0).
            5. Remove the starting node from the returned path and append the remaining nodes to the overall 
            path list.
            6. Return the final path.
        * Example Call:
        path = path_finder.path_find(["A", "B", "C"])
    """

    def path_find(self, path_list):
        path = [0]
        prev_node = 0
        path_dict = {
            "A": 2,
            "B": 6,
            "C": 11,
            "D": 9,
            "E": 16,
        }
        for i in path_list:
            p1, l1 = self.dijkstra(prev_node, path_dict[i])
            p1.pop(0)
            path.extend(p1)
            prev_node = path_dict[i]
        p, l = self.dijkstra(prev_node, 0)
        p.pop(0)
        path.extend(p)
        return path

    """
        * Function Name: gendirect
        * Input:
            * `self`: Reference to the object of the class where the function is defined. 
            It is assumed that the class has a member named `directions` which is a dictionary 
            mapping strings of three consecutive nodes in the path to their corresponding directions.
            * `path`: A list of node names representing the path.
        * Output:
            * A list containing two elements:
                * `final_path`: A list of direction instructions for the path, where "s" 
                represents "start", "u" represents "up", and other characters represent directions 
                from the `directions` dictionary.
                * `original_path`: The original path list provided as input.
        * Logic:
            1. Initialize an empty `final_path` list and add "s" to indicate the starting point.
            2. Iterate through the path list, skipping the first and last element 
            (since they don't have two neighbors).
            3. Check if the current node and its next neighbor are the same:
                * If yes, add "u" to the `final_path` to indicate going up.
                * If no, construct a key string by concatenating the current, previous, and next 
                node names.
                * Look up the corresponding direction in the `directions` dictionary and add 
                it to the `final_path`.
            4. Return a list containing the `final_path` and the original `path` for reference.

        * Example Call:
        final_path, original_path = path_finder.gendirect(["A", "B", "C", "D", "E"])

    """

    def gendirect(self, path):
        final_path = []
        final_path.append("s")
        for i in range(1, len(path) - 1):
            if path[i - 1] == path[i + 1]:
                final_path.append("u")
            else:
                k = str(path[i - 1]) + " " + str(path[i]) + " " + str(path[i + 1])
                s = self.directions[k]
                final_path.append(s)
        # print(path)
        # print(final_path)
        return [final_path, path]


if __name__ == "__main__":
    x = path_gen()
