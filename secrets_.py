from configparser import ConfigParser
import psycopg2
from psycopg2 import extras


def load_config(filename, section):
    parser = ConfigParser()
    parser.read(filename)

    # get section, default to postgresql
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return config

def api_key() :
    return '0_2MWuIiIwpeIBwEpgQDaQlYSJhMkaQw'


def config_locn() :
    return '/home/rajat/Documents/Tidal/Test_1/Launch_files/Config_files/'

def feat_builder_locn() :
    return '/home/ubuntu/PycharmProjects/Features/'

def index_file() :
    return '/home/ubuntu/Tidal/WIP/temp/SPY1minute.pkl'


def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            conn.autocommit = True
            print('Connected to the PostgreSQL server.')
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)



def connect_no_auto(config):
    """ Connect to the PostgreSQL database server """
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            conn.autocommit = False
            print('Connected to the PostgreSQL server.')
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)