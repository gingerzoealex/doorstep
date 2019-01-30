import json

class DoorstepMetadata:
    def __init__(self, docker_image=None, docker_revision=None, context_package=None, configuration={}):
        self.docker = {
            "image": docker_image,
            "revision": docker_revision
        }
        self.package = context_package
        self.configuration = configuration

    def __repr__(self):
        return "<DoorstepMetadata: {}>".format(self.to_dict())

    def has_package(self):
        return bool(self.context["package"])

    @property
    def package(self):
        package = self._context_package

        if type(package) is str:
            package = json.loads(package)
            self._context_package = package

        return package

    @package.setter
    def package(self, package):
        self._context_package = package

    @classmethod
    def from_dict(cls, dct):
        kwargs = {}

        if "docker" in dct:
            kwargs["docker_image"] = dct["docker"]["image"]
            kwargs["docker_revision"] = dct["docker"]["revision"]

        if "context" in dct:
            if "package" in dct["context"]:
                kwargs["context_package"] = dct["context"]["package"]

        if "configuration" in dct:
            kwargs["configuration"] = dct["configuration"]

        return cls(**kwargs)

    def to_dict(self):
        package = self._context_package
        if type(package) is not str:
            package = json.dumps(package)

        return {
            "docker": dict(self.docker),
            "context": {
                "package": package
            },
            "configuration": dict(self.configuration)
        }
