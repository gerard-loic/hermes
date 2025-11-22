from hermes.utils.log import Log

class CliManager:
    @staticmethod
    def cmd(args:list, main):
        Log.write("=============================================================")
        Log.write("Switch to CLI Mode")
        if len(args) == 1:
            Log.write("""
Command not specified
Supported commandes : 
train : train models
""")           
        elif args[1] == "train":
            CliManager.__cmdTrain(args, main)

    @staticmethod
    def __cmdTrain(args, main):
        if len(args) != 3:
            Log.write("""
Train option not supported.
Supported options : 
--all : train both models
--classifier : train classifier model
--ner : train Ner model
""")
      
        elif args[2] == "--all":
            main.train()
        elif args[2] == "--classifier":
            main.train(trainClassifierModel=True, trainNerModel=False)
        elif args[2] == "--ner":
            main.train(trainClassifierModel=False, trainNerModel=True)
        else:
            Log.write("""
Train option not supported.
Supported options : 
--all : train both models
--classifier : train classifier model
--ner : train Ner model
""")
            
