import time
from queue import PriorityQueue, deque
import matplotlib.pyplot as plt 
import csv
from math import sqrt, radians, sin, cos, atan2

# Implement function to read coordinates file
def city_coordinates(fname):
    city_coords = {}
    with open(fname, "r") as file:
        read = csv.reader(file)
        for row in read:
            # Read: City, Latitude, Longitude
            cname = row[0]
            latitude = float(row[1])
            longitude = float(row[2])
            city_coords[cname] = (latitude, longitude)
    return city_coords

# Implement function to create a map using the Adjanencies text file.
def map_graph(fname):
    graph = {}
    with open(fname, "r") as file:
        for line in file:
            # Read: City 1, City 2
            cities = line.strip().split()
            if len(cities) == 2:
                city1, city2 = cities

                if city1 not in graph:
                    graph[city1] = []
                if city2 not in graph:
                    graph[city2] = []

                # Connnects cities together
                graph[city1].append(city2)
                graph[city2].append(city1)
    return graph

def calculate_distance(city1, city2):
    if city1 not in city_coords or city2 not in city_coords:
        return 0

    lat1, lon1 = city_coords[city1]
    lat2, lon2 = city_coords[city2]

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371
    return R * c

# ChatGPT—Prompt: "How do I calculate the distance between two cities using a txt file that has their coordinates"

def total_route_distance(path):
    if not path or len(path) < 2:
        return 0  
    total_distance = sum(calculate_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
    return total_distance

#______________________________________________________________________
# Undirected Brute-Force Approaches
# Implementation of the Breadth-First Search
def breadthFirst_Search(start, end):
    queue = deque()
    queue.append([start])
    visited = set()

    while queue:
        path = queue.popleft()
        city = path[-1]

        if city == end:
            return path
        
        if city not in visited:
            visited.add(city)

            for neighbor in sorted(graph.get(city, [])):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append(new_path)
    return None
# Geeks for Geeks: BFS Article
# https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
        
# Implementation of Depth-First Search
def depthFirst_Search(start, end, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = [start]
    if start == end:
        return path  # Once the destination is reached, return path
    
    visited.add(start)

    # Iterate through the nodes
    for neighbor in sorted(graph.get(start, [])):
        # Check if node has been visited
        if neighbor not in visited:
            # Create and add to the path
            new_path = depthFirst_Search(neighbor, end, visited, path + [neighbor])
            if new_path:
                return new_path
    return None
# ChatGPT—Prompt: "How do you make a depth first search"
# ChatGPT—Prompt: "I would like the function to return a path"

# Implementation of ID-DFS Search
def IDDFS_Search(start, end, depth_limit=10):
    def dls(node, target, depth):
        if depth == 0:
            return None
        if node == target:
            return [node]
        for neighbor in graph.get(node, []):
            path = dls(neighbor, target, depth - 1)
            if path:
                return [node] + path
        return None
    
    for depth in range(depth_limit):
        result = dls(start, end, depth)
        if result:
            return result    
    return None
# Geeks For Geeks: IDDFS Article 
# https://www.geeksforgeeks.org/iterative-deepening-searchids-iterative-deepening-depth-first-searchiddfs/
# ChatGPT—Prompt: "How do you make a ID-DFS search function using a dls within the function in python?"

#______________________________________________________________________
# Heuristic Approaches
# Implementation of the Best-First Search
def bestFirst_Search(start, end):
    priorQueue = PriorityQueue()
    priorQueue.put((0, [start]))

    # Track visited cities 
    visited = set()

    # Iterate through if not empty
    while not priorQueue.empty():
        # Retrieve cities with the lowest priority
        _, path = priorQueue.get()
        city = path[-1]

        # If destination is reached, return path
        if city == end:
            return path
        
        # Iterates through cities not visited
        if city not in visited:
            visited.add(city)
            for neighbor in graph.get(city, []):
                priorQueue.put((len(path), path + [neighbor]))
    return None
# Geeks For Geeks: Best First Search
# https://www.geeksforgeeks.org/best-first-search-informed-search/

# Implementation of the A* search
def heuristic(city1, city2):
    if city1 not in city_coords or city2 not in city_coords:
        return float("inf")
    lat1, lon1 = city_coords[city1]
    lat2, lon2 = city_coords[city2]
    return sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def a_star_Search(start, end):
    priorQueue = PriorityQueue()
    priorQueue.put((0, [start]))
    visited = set()
    g_cost = {start: 0}

    while not priorQueue.empty():
        _, path = priorQueue.get()
        city = path[-1]

        if city == end:
            return path
        
        if city not in visited:
            visited.add(city)

            for neighbor in sorted(graph.get(city, [])):
                new_cost = g_cost[city] + 1
                h_cost = new_cost + heuristic(neighbor, end)
                f_cost = new_cost + h_cost

                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    priorQueue.put((f_cost, path + [neighbor]))
# ChatGPT—Prompt: "how do you create an A* Search"
# ChatGPT—Prompt: "how do I implement a heuristic function for A* Search"

# Present the user the options available
def receive_user_input():
    # Ask user for start and end cities
    print("These are the list of cities: ", list(graph.keys()))

    print("______________________________________________")
    print("What Search Method would you like to use?:")
    print("A. Breadth-First Search(BFS)")
    print("B. Depth-First Search(DFS)")
    print("C. Iterative Deepening DFS(ID-DFS)")
    print("D. Best-First Search")
    print("E. A* Search")
    print("Q. Quit\n")
    choices = ["A", "B", "C", "D", "E", "Q"]

    while True:
        option = input("Option: ").strip().upper()
        if option in choices:
            if option != "Q":
                print("\n\nYou have choosen option: ", option)
                
                start = input("\nEnter starting city: ")
                end = input("Enter destination city: ")

                if start not in graph or end not in graph:
                    print("ERROR: Invalid City. Try Again")
                    return receive_user_input()
                return start, end, option
        if option == "Q":
            start = ""
            end = ""
            return start, end, option
        else:
            print("ERROR: Invalid Choice. Try Again\n")
            

def plot_graph(city_coords, graph, path=None):
    plt.figure(figsize=(7, 7))

    for city, (lat, lon) in city_coords.items():
        plt.scatter(lon, lat, label=city, s=100)
        plt.text(lon + 0.01, lat + 0.01, city, fontsize=8)

    for city, neighbors in graph.items():
        lat1, lon1 = city_coords.get(city, (None, None))
        for neighbor in neighbors:
            lat2, lon2 = city_coords.get(neighbor, (None, None))
            if lat1 and lon1 and lat2 and lon2:
                plt.plot([lon1, lon2], [lat1, lat2], 'k-', lw=0.5)

    if path:
        for i in range(len(path) - 1):
            city1 = path[i]
            city2 = path[i + 1]
            lat1, lon1 = city_coords.get(city1, (None, None))
            lat2, lon2 = city_coords.get(city2, (None, None))
            if lat1 and lon1 and lat2 and lon2:
                plt.plot([lon1, lon2], [lat1, lat2], 'r-', lw=2)
                plt.scatter([lon1, lon2], [lat1, lat2], color='red', s=100)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Cities Graph")
    plt.show()
# ChatGPT—Prompt: "How do you create a plot graph using city coordinates, graph of location, and highlights the path?"

graph = map_graph("Adjacencies.txt")
city_coords = city_coordinates("coordinates.csv")

plot_graph(city_coords, graph)

start, end, option = receive_user_input()

while option != "Q":
    if option == "A":
        start_time = time.time()
        path = breadthFirst_Search(start, end)
        end_time = time.time()
    if option == "B":
        start_time = time.time()
        path = depthFirst_Search(start, end)
        end_time = time.time()
    if option == "C":
        start_time = time.time()
        path = IDDFS_Search(start, end)
        end_time = time.time()
    if option == "D":
        start_time = time.time()
        path = bestFirst_Search(start, end)
        end_time = time.time()
    if option == "E":
        start_time = time.time()
        path = a_star_Search(start, end)
        end_time = time.time()

    if path: 
        print("Route from ", start, " to ", end, ": ", " -> ".join(path))
        print("Total distance: ", total_route_distance(path))
        print("Time taken: ", end_time - start_time, "seconds")
    else:
        print("No Route Found\n")

    plot_graph(city_coords, graph, path)

    print("\n\n______________________________________________")
    start, end, option = receive_user_input()

    if option == "Q":
        quit
    else:
        continue
