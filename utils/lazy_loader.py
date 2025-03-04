class LazyLoader:
    """Lazily import modules only when needed."""
    
    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None
        
    def __getattr__(self, name):
        if self._module is None:
            self._module = __import__(self.module_name)
        return getattr(self._module, name)