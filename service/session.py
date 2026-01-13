import time


class TurnSession:
    def __init__(self, engine):
        self.engine = engine
        self.last_state = None

        # session 生命周期管理
        now = time.time()
        self.created_ts = now
        self.last_active_ts = now

    def touch(self):
        """
        被 server 调用，用于 session 保活 / GC
        """
        self.last_active_ts = time.time()

    def feed_audio(self, audio):
        """
        audio: np.ndarray(float32)
        """
        # 每次收到音频，都算活跃
        self.touch()

        result = self.engine.process(audio)
        self.last_state = result["state"]
        return result

        # # 只在状态变化时上报
        # if result["state"] != self.last_state:
        #     self.last_state = result["state"]
        #     return result

        # return None
