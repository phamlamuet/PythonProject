import hexaly.optimizer
import sys
import math


def main(instance_file, str_time_limit, output_file):
    #
    # Read instance data
    #
    nb_trucks_per_depot, nb_customers, nb_depots, route_duration_capacity_data, \
        truck_capacity_data, demands_data, service_time_data, \
        distance_matrix_customers_data, distance_warehouse_data = read_input_mdvrp(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customer_sequences = [[model.list(nb_customers) for _ in range(nb_trucks_per_depot)] for _ in range(nb_depots)]

        # Vectorization for partition constraint
        customer_sequences_constraint = [customer_sequences[d][k]
                                         for d in range(nb_depots) for k in range(nb_trucks_per_depot)]

        # All customers must be visited by exactly one truck
        model.constraint(model.partition(customer_sequences_constraint))

        # Create Hexaly arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        service_time = model.array(service_time_data)
        dist_customers = model.array(distance_matrix_customers_data)

        # Distances traveled by each truck from each depot
        route_distances = [[None for _ in range(nb_trucks_per_depot)] for _ in range(nb_depots)]

        # Total distance traveled
        total_distance = model.sum()

        for d in range(nb_depots):
            dist_depot = model.array(distance_warehouse_data[d])
            for k in range(nb_trucks_per_depot):
                sequence = customer_sequences[d][k]
                c = model.count(sequence)

                # The quantity needed in each route must not exceed the truck capacity
                demand_lambda = model.lambda_function(lambda j: demands[j])
                route_quantity = model.sum(sequence, demand_lambda)
                model.constraint(route_quantity <= truck_capacity_data[d])

                # Distance traveled by truck k of depot d
                dist_lambda = model.lambda_function(lambda i: model.at(dist_customers, sequence[i - 1], sequence[i]))
                route_distances[d][k] = model.sum(model.range(1, c), dist_lambda) \
                    + model.iif(c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0)

                # We add service Time
                service_lambda = model.lambda_function(lambda j: service_time[j])
                route_service_time = model.sum(sequence, service_lambda)

                total_distance.add_operand(route_distances[d][k])

                # The total distance should not exceed the duration capacity of the truck
                # (only if we define such a capacity)
                if (route_duration_capacity_data[d] > 0):
                    model.constraint(route_distances[d][k] + route_service_time <= route_duration_capacity_data[d])

        # Objective: minimize the total distance traveled
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - instance, time_limit, total distance
        #  - for each depot and for each truck in this depot, the customers visited
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("Instance: " + instance_file + " ; " + "time_limit: " + str_time_limit + " ; " +
                        "Objective value: " + str(total_distance.value))
                f.write("\n")
                for d in range(nb_depots):
                    trucks_used = []
                    for k in range(nb_trucks_per_depot):
                        if (len(customer_sequences[d][k].value) > 0):
                            trucks_used.append(k)
                    if len(trucks_used) > 0:
                        f.write("Depot " + str(d + 1) + "\n")
                        for k in range(len(trucks_used)):
                            f.write("Truck " + str(k + 1) + " : ")
                            customers_collection = customer_sequences[d][trucks_used[k]].value
                            for p in range(len(customers_collection)):
                                f.write(str(customers_collection[p] + 1) + " ")
                            f.write("\n")
                        f.write("\n")


# Input files following "Cordeau"'s format
def read_input_mdvrp(filename):
    with open(filename) as f:
        instance = f.readlines()

    nb_line = 0
    datas = instance[nb_line].split()

    # Numbers of trucks per depot, customers and depots
    nb_trucks_per_depot = int(datas[1])
    nb_customers = int(datas[2])
    nb_depots = int(datas[3])

    route_duration_capacity = [None]*nb_depots  # Time capacity for every type of truck from every depot
    truck_capacity = [None]*nb_depots  # Capacity for every type of truck from every depot

    for d in range(nb_depots):
        nb_line += 1
        capacities = instance[nb_line].split()

        route_duration_capacity[d] = int(capacities[0])
        truck_capacity[d] = int(capacities[1])

    # Coordinates X and Y, service time and demand for customers
    nodes_xy = [[None, None]] * nb_customers
    service_time = [None] * nb_customers
    demands = [None] * nb_customers

    for n in range(nb_customers):
        nb_line += 1
        customer = instance[nb_line].split()

        nodes_xy[n] = [float(customer[1]), float(customer[2])]

        service_time[n] = int(customer[3])
        demands[n] = int(customer[4])

    # Coordinates X and Y of every depot
    depot_xy = [None] * nb_depots

    for d in range(nb_depots):
        nb_line += 1
        depot = instance[nb_line].split()

        depot_xy[d] = [float(depot[1]), float(depot[2])]

    # Compute the distance matrices
    distance_matrix_customers = compute_distance_matrix_customers(nodes_xy)
    distance_warehouse = compute_distance_warehouse(depot_xy, nodes_xy)

    return nb_trucks_per_depot, nb_customers, nb_depots, route_duration_capacity, \
        truck_capacity, demands, service_time, distance_matrix_customers, distance_warehouse


# Compute the distance matrix for customers
def compute_distance_matrix_customers(nodes_xy):
    nb_customers = len(nodes_xy)
    distance_matrix = [[0 for _ in range(nb_customers)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        for j in range(i+1, nb_customers):
            distij = compute_dist(nodes_xy[i], nodes_xy[j])
            distance_matrix[i][j] = distij
            distance_matrix[j][i] = distij
    return distance_matrix


# Compute the distance matrix for warehouses/depots
def compute_distance_warehouse(depot_xy, nodes_xy):
    nb_customers = len(nodes_xy)
    nb_depots = len(depot_xy)
    distance_warehouse = [[0 for _ in range(nb_customers)] for _ in range(nb_depots)]

    for i in range(nb_customers):
        for d in range(nb_depots):
            distance_warehouse[d][i] = compute_dist(depot_xy[d], nodes_xy[i])

    return distance_warehouse


# Compute the distance between two points
def compute_dist(p, q):
    return math.sqrt(math.pow(p[0] - q[0], 2) + math.pow(p[1] - q[1], 2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python mdvrp.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)