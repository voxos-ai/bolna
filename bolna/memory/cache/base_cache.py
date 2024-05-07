class BaseCache:
    def set(self, key, value, **kwargs):
        raise NotImplementedError

    def get(self, key, **kwargs):
        raise NotImplementedError
