
class Error(Exception):
    """Clase base para otra excepcións"""
    pass

class DatasetMixing(Error):
    pass

class NoSpeechDetected(Error):
    pass

class LimitTooSmall(Error):
    pass

class FeatureExtractionFail(Error):
    pass

class ExitApp(Error):
    pass
