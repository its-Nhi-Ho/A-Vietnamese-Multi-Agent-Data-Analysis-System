"""
data_store.py - Nơi lưu các dataset (DataFrame) đang "sống" trong session, dùng chung
cho mọi tool/agent.
"""


class DataStore:
    """Singleton lưu các dataset đang active trong session."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.datasets = {}
            cls._instance.history = []
        return cls._instance

    def add(self, name, df):
        self.datasets[name] = df

    def get(self, name):
        if name not in self.datasets:
            raise ValueError(
                f"Dataset '{name}' không tồn tại. Hiện có: {list(self.datasets.keys())}"
            )
        return self.datasets[name]

    def log(self, action):
        self.history.append(action)

    def debug(self) -> None:
        """In thực tế dataset nào ĐANG có trong store (độc lập với những gì agent 'nói'
        nó đã làm) - hữu ích để phát hiện hallucination."""
        if not self.datasets:
            print("⚠️ store.datasets RỖNG - chưa có dataset nào thực sự được load/tạo.")
            return
        for name, df in self.datasets.items():
            print(f"- '{name}': {df.shape[0]} dòng x {df.shape[1]} cột -> {list(df.columns)}")


store = DataStore()
