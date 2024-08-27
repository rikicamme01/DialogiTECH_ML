import neptune.new as neptune

class NeptuneLogger():
    def __init__(self) -> None:
        #Neptune initialization
        self.run = neptune.init(
            project="Insert Neptune project",
            api_token="Insert your API token",
        )
    