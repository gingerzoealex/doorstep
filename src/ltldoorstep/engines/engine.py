from contextlib import contextmanager

class Engine:
    def __init__(self, config=None):
        pass

    @staticmethod
    def description():
        return '(not provided)'

    @staticmethod
    def config_help():
        return None

    def add_data(self, filename, content, redirect, session):
        raise NotImplementedError("Function must be implemented")

    def add_processor(self, modules, metadata, session):
        raise NotImplementedError("Function must be implemented")

    async def run(self, filename, workflow_module, metadata, bucket=None):
        raise NotImplementedError("Function must be implemented")

    async def monitor_pipeline(self, session):
        raise NotImplementedError("Function must be implemented")

    async def get_output(self, session):
        raise NotImplementedError("Function must be implemented")

    @contextmanager
    def make_session(self):
        raise NotImplementedError("Function must be implemented")
