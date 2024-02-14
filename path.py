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
    print(x.gendirect(x.path_find(["E", "B", "A", "D", "C"])))
