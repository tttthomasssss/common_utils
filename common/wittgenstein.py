__author__ = 'thk22'
from importlib import import_module
import collections
'''
Who's your favourite Metaphysician?
'''


def create_instance(fully_qualified_name, **kwargs):
	module_name, class_name = fully_qualified_name.rsplit('.', 1)
	module = import_module(module_name)

	class_ = getattr(module, class_name)
	instance = class_(**kwargs)

	return instance


def create_function(fully_qualified_name):
	if (isinstance(fully_qualified_name, collections.Callable)):
		return fully_qualified_name

	module_name, function_name = fully_qualified_name.rsplit('.', 1)
	module = import_module(module_name)

	fn_ = getattr(module, function_name)

	return fn_


def prepare_invocation_on_obj(obj, function_name):
	fn_ = getattr(obj, function_name)

	return fn_


def getattr_from_module(fully_qualified_name, attr):
	module = import_module(fully_qualified_name)

	attr_ = getattr(module, attr)

	return attr_