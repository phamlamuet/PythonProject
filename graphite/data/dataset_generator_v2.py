# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Defines class for create a dataset based on the synthetic distribution
from abc import ABC, abstractmethod
from typing import List, Dict
from graphite.protocol import GraphV2Problem, GraphV2ProblemMulti, MAX_SALESMEN
import os
import random
import json
import numpy as np
from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("dataset")

class DatasetGenerator(ABC):
    @classmethod
    @abstractmethod
    def generate_n_samples(cls, n:int, problem_model:GraphV2Problem, loaded_datasets, **kwargs)->List[GraphV2Problem]:
        '''
        Returns a list of dictionaries containing the parameters required to initialize each problem
        This can be used in-situ to generate a dataset without saving it
        '''
        ...

    @classmethod
    def generate_and_save_dataset(cls, n_samples: int, file_name: str=None, save_dir: str=None, loaded_datasets: Dict={}):
        '''
        generates n sample problems and saves it as a json file
        '''
        if save_dir is None:
            save_dir = cls.save_dir
        if file_name is None:
            file_name = cls.file_name
        
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate samples - Note that this these samples are generated with out edges
        problems = cls.generate_n_samples(n_samples, loaded_datasets)
        
        # serialize and save json string with meta data of the dataset
        output_json = cls.serialize_dataset(problems)
        with open(os.path.join(save_dir, file_name), 'w') as f:
            f.write(output_json)

        return problems
    
    @classmethod
    def generate_and_save_dataset_without_edges(cls, n_samples: int, file_name: str=None, save_dir: str=None, loaded_datasets: Dict={}):
        '''
        generates n sample GraphV2Problems and saves it as a json file
        '''
        if save_dir is None:
            save_dir = cls.save_dir
        if file_name is None:
            file_name = cls.file_name
        
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate samples - Note that this these samples are generated with out edges
        problems, problem_sizes = cls.generate_n_samples_without_edges(n_samples, loaded_datasets)
        
        # serialize and save json string with meta data of the dataset
        output_json = cls.serialize_dataset(problems)
        with open(os.path.join(save_dir, file_name), 'w') as f:
            f.write(output_json)
        
        return problems
    
    @classmethod
    def serialize_dataset(cls, problems: List[GraphV2Problem]):
        meta_data = {
            'problem_type': 'Metric TSP',
            'n_samples': len(problems),
            'problems': [problem.model_dump_json() for problem in problems],
        }
        return json.dumps(meta_data)
    
    @classmethod
    def load_dataset(cls, problem_class, filepath=None):
        if filepath == None:
            filepath = os.path.join(cls.save_dir, cls.file_name)
        with open(filepath,'r') as f:
            raw_data = json.load(f)
        
        # rebuild the problems given a problem_class
        dataset = raw_data.copy()
        dataset['problems'] = [problem_class(**json.loads(data)) for data in dataset['problems']]
        return dataset
    

class MetricTSPV2Generator(DatasetGenerator):
    save_dir = os.path.join(DATASET_DIR,'metric_tsp_v2')
    file_name = os.path.join('dataset.json')

    @classmethod
    def _problem_size(self, problem: GraphV2Problem):
        return problem.n_nodes // 1000 * 1000 # round problem size to the nearest 1000 (just for categorical assignment)
    
    @classmethod
    def recreate_edges(cls, problem: GraphV2Problem, loaded_datasets):
        node_coords_np = loaded_datasets[problem.dataset_ref]["data"]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            problem.edges = geom_edges(node_coords).tolist()
        elif problem.cost_function == "Euclidean2D":
            problem.edges = euc_2d_edges(node_coords).tolist()
        elif problem.cost_function == "Manhatten2D":
            problem.edges = man_2d_edges(node_coords).tolist()
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
        
    @classmethod
    def generate_n_samples(cls, n: int, loaded_datasets):
        '''
        This method generates n_samples with edges recreated. This is very intensive on the memory consumption so it is advised to keep n small.
        '''
        problems = []
        for _ in range(n):
            n_nodes = random.randint(2000, 5000)
            # randomly select n_nodes indexes from the selected graph
            dataset_ref = random.sample(list(loaded_datasets.keys()),1)[0]
            selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]["data"])), n_nodes)
            test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)
            cls.recreate_edges(test_problem, loaded_datasets)
            problems.append(test_problem)
        return problems, [cls._problem_size(problem) for problem in problems]

    @classmethod
    def generate_n_samples_without_edges(cls, n: int, loaded_datasets):
        problems = []
        for _ in range(n):
            n_nodes = random.randint(2000, 5000)
            # randomly select n_nodes indexes from the selected graph
            dataset_ref = random.sample(list(loaded_datasets.keys()),1)[0]
            selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]["data"])), n_nodes)
            test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)
            problems.append(test_problem)
        return problems, [cls._problem_size(problem) for problem in problems]
    
    @classmethod
    def generate_one_sample(cls, n_nodes:int, loaded_datasets):
        dataset_ref = random.sample(list(loaded_datasets.keys()),1)[0]
        selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]["data"])), n_nodes)
        test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)
        cls.recreate_edges(test_problem, loaded_datasets)
        return test_problem
    
class MetricMTSPV2Generator(DatasetGenerator):
    save_dir = os.path.join(DATASET_DIR,'mtsp_v2')
    file_name = os.path.join('dataset.json')

    @classmethod
    def _problem_size(self, problem: GraphV2ProblemMulti):
        return problem.n_nodes // 500 * 500 # round down problem size to the nearest 500
    
    @classmethod
    def recreate_edges(cls, problem: GraphV2ProblemMulti, loaded_datasets):
        node_coords_np = loaded_datasets[problem.dataset_ref]["data"]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            problem.edges = geom_edges(node_coords).tolist()
        elif problem.cost_function == "Euclidean2D":
            problem.edges = euc_2d_edges(node_coords).tolist()
        elif problem.cost_function == "Manhatten2D":
            problem.edges = man_2d_edges(node_coords).tolist()
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
        
    @classmethod
    def generate_n_samples(cls, n: int, loaded_datasets):
        '''
        This method generates n_samples with edges recreated. This is very intensive on the memory consumption so it is advised to keep n small.
        '''
        problems = []
        single_depot_choices = [random.choice([True, False]) for _ in range(n)]
        for i in range(n):
            n_nodes = random.randint(500, 2000)
            n_salesmen = random.randint(2, MAX_SALESMEN)
            # randomly select n_nodes indexes from the selected graph
            dataset_ref = random.sample(list(loaded_datasets.keys()),1)[0]
            selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]["data"])), n_nodes)
            if single_depot_choices[i]:
                test_problem = GraphV2ProblemMulti(problem_type="Metric mTSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref, n_salesmen=n_salesmen, depots=[0 for _ in range(n_salesmen)])
            else:
                test_problem = GraphV2ProblemMulti(problem_type="Metric mTSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref, n_salesmen=n_salesmen, single_depot=False, depots=random.sample(range(n_nodes), k=n_salesmen))
            cls.recreate_edges(test_problem, loaded_datasets)
            problems.append(test_problem)
        return problems, [cls._problem_size(problem) for problem in problems]

    @classmethod
    def generate_n_samples_without_edges(cls, n: int, loaded_datasets):
        problems = []
        single_depot_choices = [random.choice([True, False]) for _ in range(n)]
        for i in range(n):
            n_nodes = random.randint(500, 2000)
            n_salesmen = random.randint(2, MAX_SALESMEN)
            # randomly select n_nodes indexes from the selected graph
            dataset_ref = random.sample(list(loaded_datasets.keys()),1)[0]
            selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]["data"])), n_nodes)
            if single_depot_choices[i]:
                test_problem = GraphV2ProblemMulti(problem_type="Metric mTSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref, n_salesmen=n_salesmen, depots=[0 for _ in range(n_salesmen)])
            else:
                test_problem = GraphV2ProblemMulti(problem_type="Metric mTSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref, n_salesmen=n_salesmen, single_depot=False, depots=random.sample(range(n_nodes), k=n_salesmen))
            problems.append(test_problem)
        return problems, [cls._problem_size(problem) for problem in problems]
    
    @classmethod
    def generate_one_sample(cls, loaded_datasets, n_nodes:int=None, n_salesmen:int=None, single_depot:bool=True):
        dataset_ref = random.sample(list(loaded_datasets.keys()),1)[0]
        if not n_nodes:
            n_nodes = random.randint(500, 2000)
        if not n_salesmen:
            n_salesmen = random.randint(2, MAX_SALESMEN)
        selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]["data"])), n_nodes)
        if single_depot:
            test_problem = GraphV2ProblemMulti(problem_type="Metric mTSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref, n_salesmen=n_salesmen, depots=[0 for _ in range(n_salesmen)])
        else:
            test_problem = GraphV2ProblemMulti(problem_type="Metric mTSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref, n_salesmen=n_salesmen, single_depot=False, depots=random.sample(range(n_nodes), k=n_salesmen))
        cls.recreate_edges(test_problem, loaded_datasets)
        return test_problem

