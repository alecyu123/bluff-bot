import os
import csv

class Logger(object):
    ''' Logger saves the running results and helps make plots from the results
    '''

    def __init__(self, log_dir):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        '''
        self.log_dir = log_dir

    def __enter__(self):
        self.txt_path = os.path.join(self.log_dir, 'log.txt')
        self.csv_path = os.path.join(self.log_dir, 'performance.csv')
        self.fig_path = os.path.join(self.log_dir, 'fig.png')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        fieldnames = ['episode', 'reward']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        return self

    def log(self, text):
        ''' Write the text to log file then print it.
        Args:
            text(string): text to log
        '''
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, episode, reward):
        ''' Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'episode': episode, 'reward': reward})
        print('')
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward))
        self.log('----------------------------------------')

    def __exit__(self, type, value, traceback):
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()
        print('\nLogs saved in', self.log_dir)

class MultiLogger:
    ''' Manages multiple Logger instances for different agents or metrics. '''
    def __init__(self, base_log_dir, logger_keys):
        '''
        Args:
            base_log_dir (str): The main directory to store all logs.
            logger_keys (list): A list of strings identifying each logger (e.g., ['DQN', 'VAEDQN']).
        '''
        self.base_log_dir = base_log_dir
        self.logger_keys = logger_keys
        self.loggers = {}
        self._logger_paths = {} # To store paths after initialization

        # Create base directory
        os.makedirs(self.base_log_dir, exist_ok=True)

        # Setup individual loggers
        for key in self.logger_keys:
            key_log_dir = os.path.join(self.base_log_dir, f"{key.lower()}_log")
            self.loggers[key] = Logger(key_log_dir)

    def __enter__(self):
        '''Enter context for all managed loggers.'''
        for key in self.loggers:
            self.loggers[key].__enter__()
            # Store paths after the sub-logger's __enter__ is called
            self._logger_paths[key] = {
                'csv': self.loggers[key].csv_path,
                'fig': self.loggers[key].fig_path,
                'txt': self.loggers[key].txt_path
            }
        return self

    def __exit__(self, type, value, traceback):
        '''Exit context for all managed loggers.'''
        exceptions = []
        for key in self.loggers:
            try:
                self.loggers[key].__exit__(type, value, traceback)
            except Exception as e:
                exceptions.append(e)
        print(f'\nAll logs saved in base directory: {self.base_log_dir}')
        if exceptions:
            # Optionally re-raise or handle exceptions from sub-loggers
            print("Exceptions occurred during logger exit:", exceptions)


    def log(self, key, text):
        ''' Log text to a specific logger's text file. '''
        if key in self.loggers:
            self.loggers[key].log(text)
        else:
            print(f"Warning: Logger key '{key}' not found for logging text.")

    def log_performance(self, key, episode, reward):
        ''' Log performance to a specific logger's csv file. '''
        if key in self.loggers:
            self.loggers[key].log_performance(episode, reward)
        else:
            print(f"Warning: Logger key '{key}' not found for logging performance.")

    def get_paths(self, key):
        ''' Get the CSV and Figure paths for a specific logger key. '''
        if key in self._logger_paths:
            return self._logger_paths[key]['csv'], self._logger_paths[key]['fig']
        else:
            # Fallback: construct expected path (less reliable if Logger changes internal names)
            key_log_dir = os.path.join(self.base_log_dir, f"{key.lower()}_log")
            csv_p = os.path.join(key_log_dir, 'performance.csv')
            fig_p = os.path.join(key_log_dir, 'fig.png')
            print(f"Warning: Paths for logger '{key}' not found in cache, constructing expected paths.")
            return csv_p, fig_p

    def close(self):
        ''' Explicitly close all managed loggers. '''
        self.__exit__(None, None, None)