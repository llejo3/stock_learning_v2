import os


class WindowUtils:

    @staticmethod
    def logout():
        return os.system("shutdown -l")

    @staticmethod
    def restart():
        return os.system("shutdown /r /t 1")

    @staticmethod
    def shutdown():
        return os.system("shutdown /s /t 1")
