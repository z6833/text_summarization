# coding=utf-8
import logging

logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='#%Y-%m-%d %H:%M:%S')

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()
