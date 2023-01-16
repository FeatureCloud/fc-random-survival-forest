import pickle
import shutil
import threading
import time

import joblib
import json
import jsonpickle
import pandas as pd
import yaml

from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, State
from algo import Coordinator, Client

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """
    
    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        print("Initializing")
        if self.id is not None:  # Test if setup has happened already
            print("Coordinator", self.is_coordinator)
            if self.is_coordinator:
                self.store('client', Coordinator())
            else:
                self.store('client', Client())
            self.store('time', time.time())
        return 'read input'


@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """
    
    def register(self):
        self.register_transition('local computation', Role.COORDINATOR)
        self.register_transition('wait coordinator input', Role.PARTICIPANT)
        self.register_transition('read input', Role.BOTH)

    def run(self) -> str or None:
        try:
            print('[CLIENT] Read input and config')
            self.read_config()
            if self.is_coordinator:
                data_to_broadcast = json.dumps({'event_column': self.load('event_column'),
                    'time_column': self.load('dur_column'),
                    'n_estimators_local': self.load('n_estimators_local'),
                    'min_sample_leafes': self.load('min_sample_leafes'),
                    'min_sample_split': self.load('min_sample_split'),
                    'iterations_fi': self.load('iterations_fi'),
                    'min_concordant_pairs': self.load('min_concordant_pairs'),
                    'random_state': self.load('random_state'),
                    'merge_test_train': self.load('merge_test_train')
                })
                self.broadcast_data(data_to_broadcast, send_to_self=False)
                
                print(f"minsample leafs: {self.load('min_sample_leafes')}")
                return 'local computation'
            else:
                return 'wait coordinator input'
        
        except Exception as e:
            self.log('no config file or missing fields', LogLevel.ERROR)
            self.update(message='no config file or missing fields', state=State.ERROR)
            print(e)
            return 'read input'
            
    def read_config(self):
        self.store('INPUT_DIR', "/mnt/input")
        self.store('OUTPUT_DIR', "/mnt/output")
        with open(self.load('INPUT_DIR') + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_rsf']
            self.store('input', config['files']['input'])
            self.store('input_test', config['files']['input_test'])
            self.store('data', pd.read_csv(self.load('INPUT_DIR') + "/" + self.load('input')))
            self.store('data_test', pd.read_csv(self.load('INPUT_DIR') + "/" + self.load('input_test')))

            if self.is_coordinator:
                self.store('dur_column', config['parameters']['time_column'])
                self.store('event_column', config['parameters']['event_column'])
                self.store('n_estimators_local', config['parameters']['n_estimators_local'])
                self.store('min_sample_leafes', config['parameters']['min_sample_leafes'])
                self.store('min_sample_split', config['parameters']['min_sample_split'])
                self.store('iterations_fi', config['parameters']['iterations_fi'])
                self.store('min_concordant_pairs', config['parameters']['min_concordant_pairs'])
                self.store('random_state', config['parameters']['random_state'])
                self.store('merge_test_train', config['parameters']['merge_test_train'])

                print(f"minsample leafs: {self.load('min_sample_leafes')}")

        shutil.copyfile(self.load('INPUT_DIR') + '/config.yml', self.load('OUTPUT_DIR') + '/config.yml')
        print(f'Read config file.', flush=True)
 
 
@app_state('wait coordinator input', Role.PARTICIPANT)
class WaitCoordinatorInputState(AppState):
    """
    The client waits until he gets the input parameters for the computation from the coordinator.
    """
    
    def register(self):
        self.register_transition('local computation', Role.PARTICIPANT)
        self.register_transition('wait coordinator input', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        try:
            data = self.await_data()
            print('[CLIENT] Get input from coordinator')
            config_data = json.loads(data)
            self.store('event_column', config_data['event_column'])
            self.store('dur_column', config_data['time_column'])
            self.store('n_estimators_local', config_data['n_estimators_local'])
            self.store('min_sample_leafes', config_data['min_sample_leafes'])
            self.store('min_sample_split', config_data['min_sample_split'])
            self.store('iterations_fi', config_data['iterations_fi'])
            self.store('min_concordant_pairs', config_data['min_concordant_pairs'])
            self.store('random_state', config_data['random_state'])
            self.store('merge_test_train', config_data['merge_test_train'])
            print(f"coordinator input: \n"
                  f"event_column: {self.load('event_column')} \n"
                  f"dur_column: {self.load('dur_column')} \n"
                  f"n_estimators_local: {self.load('n_estimators_local')} \n"
                  f"min_sample_leafes: {self.load('min_sample_leafes')} \n"
                  f"min_sample_split: {self.load('min_sample_split')} \n"
                  f"iterations_fi: {self.load('iterations_fi')} \n"
                  f"min_concordant_pairs: {self.load('min_concordant_pairs')} \n"
                  f"random_state: {self.load('random_state')} \n"
                  f"merge_test_train: {self.load('merge_test_train')} \n"
                  )

            return 'local computation'
 
        except Exception as e:
            self.log('error wait coordinator input', LogLevel.ERROR)
            self.update(message='error wait coordinator input', state=State.ERROR)
            print(e)
            return 'wait coordinator input'
            
            
@app_state('local computation', Role.BOTH)
class LocalComputationState(AppState):
    """
    Perform the local computation of the RSF.
    """
    
    def register(self):
        self.register_transition('local computation', Role.BOTH)
        self.register_transition('global aggregation', Role.COORDINATOR)
        self.register_transition('wait for aggregation', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        try:
            print("[CLIENT] Perform local computation")
            client = self.load('client')
            rsf, Xt, y, X_test, y_test, features, concordant_pairs, actual_concordant_pairs, train_samples, test_samples = \
                client.calculate_local_rsf(self.load('data'), self.load('data_test'), self.load('dur_column'), self.load('event_column'),
                                            self.load('n_estimators_local'), self.load('min_sample_leafes'),self.load('min_sample_split'),
                                            self.load('min_concordant_pairs'), self.load('merge_test_train'), self.load('random_state'))

            self.store('X', Xt)
            self.store('y', y)
            self.store('X_test', X_test)
            self.store('y_test', y_test)
            self.store('features', features)
            self.store('concordant_pairs', concordant_pairs)
            self.store('actual_concordant_pairs', actual_concordant_pairs)
            self.store('train_samples', train_samples)
            self.store('test_samples', test_samples)
            
            data_to_send = jsonpickle.encode(rsf)
            self.send_data_to_coordinator(data_to_send)
            print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if self.is_coordinator:
                return 'global aggregation'
            else:
                return 'wait for aggregation'
            
        except Exception as e:
            self.log('error local computation', LogLevel.ERROR)
            self.update(message='error local computation', state=State.ERROR)
            print(e)
            return 'local computation'


@app_state('wait for aggregation', Role.PARTICIPANT)
class WaitForAggregationState(AppState):
    """
    Wait for the aggregation result.
    """
    
    def register(self):
        self.register_transition('wait for aggregation', Role.PARTICIPANT)
        self.register_transition('evaluation of global model', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        try:
            print("[CLIENT] Wait for aggregation")
            data = self.await_data()
            print("[CLIENT] Received aggregation data from coordinator.")
            global_rsf = jsonpickle.decode(data)
            client = self.load('client')
            client.set_global_rsf(global_rsf)
            self.store('global_rsf', global_rsf)
            return 'evaluation of global model'
        
        except Exception as e:
            self.log('error wait for aggregation', LogLevel.ERROR)
            self.update(message='error wait for aggregation', state=State.ERROR)
            print(e)
            return 'wait for aggregation'


@app_state('evaluation of global model', Role.BOTH)
class EvaluationOfGlobalModelState(AppState):
    """
    Evaluate the global model on local test data.
    """
    
    def register(self):
        self.register_transition('waiting for evaluation', Role.PARTICIPANT)
        self.register_transition('aggregation of evaluation', Role.COORDINATOR)
        self.register_transition('evaluation of global model', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            global_rsf_pickled = jsonpickle.encode(self.load('global_rsf'))
            client = self.load('client')
            ev_result = client.evaluate_global_model_with_local_test_data(global_rsf_pickled, self.load('X_test'),
                                                                               self.load('y_test'), self.load('features'),
                                                                               self.load('concordant_pairs'), self.load('iterations_fi'))

            self.store('cindex_on_global_model', ev_result[0][0])
            self.store('feature_importance_on_global_model', ev_result[1])
            
            data_to_send = pickle.dumps(ev_result)
            self.send_data_to_coordinator(data_to_send)
            print(f'[CLIENT] Sending EVALUATION data to coordinator', flush=True)

            if self.is_coordinator:
                return 'aggregation of evaluation'
            else:
                return 'waiting for evaluation'
        
        except Exception as e:
            self.log('error evaluation of global model', LogLevel.ERROR)
            self.update(message='error evaluation of global model', state=State.ERROR)
            print(e)
            return 'evaluation of global model'


@app_state('waiting for evaluation', Role.PARTICIPANT)
class WaitingForEvaluationState(AppState):
    """
    Wait for the aggregted results of the evaluation.
    """
    
    def register(self):
        self.register_transition('write results', Role.PARTICIPANT)
        self.register_transition('waiting for evaluation', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        try:
            print("[CLIENT] Wait for EVALUATION aggregation")
            data = self.await_data()
            print("[CLIENT] Received EVALUATION aggregation data from coordinator.")
            diff_c_index = jsonpickle.decode(data)
            self.store('global_c_index', diff_c_index[0])
            self.store('global_c_index_concordant_pairs', diff_c_index[1])
            print(f"global cindex: {self.load('global_c_index')}")
            print(f"global cindex concordant: {self.load('global_c_index_concordant_pairs')}")
            return 'write results'
        
        except Exception as e:
            self.log('error waiting for evaluation', LogLevel.ERROR)
            self.update(message='error waiting for evaluation', state=State.ERROR)
            print(e)
            return 'waiting for evaluation'

# GLOBAL PART
@app_state('global aggregation', Role.COORDINATOR)
class GlobalAggregationState(AppState):
    """
    Coordinator performs the aggregation of all local models into a global model.
    """
    
    def register(self):
        self.register_transition('evaluation of global model', Role.COORDINATOR)
        self.register_transition('global aggregation', Role.COORDINATOR)
        
    def run(self) -> str or None:
        try:
            print("[CLIENT] Global computation")
            data = self.gather_data()
            local_rsf_of_all_clients = [jsonpickle.decode(client_data) for client_data in data]
            client = self.load('client')
            aggregated_rsf = client.calculate_global_rsf(local_rsf_of_all_clients)
            self.store('global_rsf', aggregated_rsf)
            data_to_broadcast = jsonpickle.encode(aggregated_rsf)
            self.broadcast_data(data_to_broadcast, send_to_self=False)
            print(f'[CLIENT] Broadcasting computation data to clients', flush=True)
            return 'evaluation of global model'

        except Exception as e:
            self.log('error global aggregation', LogLevel.ERROR)
            self.update(message='error global aggregation', state=State.ERROR)
            print(e)
            return 'global aggregation'
       
       
@app_state('aggregation of evaluation', Role.COORDINATOR) #name
class AggregationOfEvaluationState(AppState):
    """
    Coordinator performs the aggregation of all local evaluation results into one evaluation result.
    """
    
    def register(self):
        self.register_transition('write results', Role.COORDINATOR)
        self.register_transition('aggregation of evaluation', Role.COORDINATOR)
        
    def run(self) -> str or None:
        try:
            print("[CLIENT] Global evaluation")
            data = self.gather_data()
            local_ev_of_all_clients = [pickle.loads(client_data) for client_data in data]
            local_c_of_all_clients = []
            tuple_c_conc = []
            for i in local_ev_of_all_clients:
                local_c_of_all_clients.append(i[0][0])
                if i[0][1] != 0:
                    tuple_c_conc.append(i[0])
                else:
                    print("We are not working with this client for evaluation! test set to small!")
            client = self.load('client')
            aggregated_c = client.calculate_global_c_index(local_c_of_all_clients)
            aggregated_c_with_conc = client.calculate_global_c_index_with_concordant_pairs(tuple_c_conc)
            self.store('global_c_index', aggregated_c)
            self.store('global_c_index_concordant_pairs', aggregated_c_with_conc)
            data_to_broadcast = jsonpickle.encode([aggregated_c, aggregated_c_with_conc])
            self.broadcast_data(data_to_broadcast, send_to_self=False)
            print(f'[CLIENT] Broadcasting EVALUATION data to clients', flush=True)
   
            return 'write results'
        
        except Exception as e:
            self.log('error aggregation of evaluation', LogLevel.ERROR)
            self.update(message='error aggregation of evaluation', state=State.ERROR)
            print(e)
            return 'aggregation of evaluation'


@app_state('write results', Role.BOTH)
class WriteresultsState(AppState):
    """
    Writes the results of global_rsf to the output_directory.
    """
    
    def register(self):
        self.register_transition('finishing', Role.COORDINATOR)
        self.register_transition('terminal', Role.PARTICIPANT)
        self.register_transition('write results', Role.BOTH)
        
    def run(self) -> str or None:
        print("[CLIENT] Writing results")
        try:
            print("[IO] Write results to output folder:")
            file_write = open(self.load('OUTPUT_DIR') + '/evaluation_result.csv', 'x')
            file_write.write("cindex_on_global_model, global_c_index_mean, global_c_index_weigthed, "
                             "training_samples, test_samples, concordant_pairs\n")
            file_write.write(f"{self.load('cindex_on_global_model')},{self.load('global_c_index')},"
                             f"{self.load('global_c_index_concordant_pairs')},{self.load('train_samples')},"
                             f"{self.load('test_samples')},{self.load('actual_concordant_pairs')}")
            file_write.close()

            with open(self.load('OUTPUT_DIR') + '/global_model.pickle', 'wb') as handle:
                pickle.dump(self.load('global_rsf'), handle, protocol=pickle.HIGHEST_PROTOCOL)

            # feature importance
            fi = self.load('feature_importance_on_global_model')
            fi.to_csv(self.load('OUTPUT_DIR') + '/feature_importance.csv', index=False)

            file_read = open(self.load('OUTPUT_DIR') + '/evaluation_result.csv', 'r')
            content = file_read.read()
            print(content)
            file_read.close()
            
            self.send_data_to_coordinator('DONE')
            
            if self.is_coordinator:
                return 'finishing'
            else:
                return 'terminal'
            
        except Exception as e:
            self.log('error write results', LogLevel.ERROR)
            self.update(message='error write results', state=State.ERROR)
            print(e)
            return 'write results'


@app_state('finishing', Role.COORDINATOR)
class FinishingState(AppState):
    
    def register(self):
        self.register_transition('finishing', Role.COORDINATOR)
        self.register_transition('terminal', Role.COORDINATOR)
        
    def run(self) -> str or None:
        try:
            print("[CLIENT] Finishing")
            self.gather_data()
            return 'terminal'

        except Exception as e:
            self.log('error finishing', LogLevel.ERROR)
            self.update(message='error finishing', state=State.ERROR)
            print(e)
            return 'finishing'
