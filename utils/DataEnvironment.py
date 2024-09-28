import os
import shutil

def clear_directory(directory_path):
    result = {'status': 'success', 'message': ''}
    try:
        shutil.rmtree(directory_path)
        os.makedirs(directory_path)        
    except Exception as e:
        result['status']='error'
        result['message']=f"Error occurred while trying to clear directory {directory_path}: {str(e)}"
    return result

class DataEnvironment:    
    def __init__(self, args,valid_lucc_info):   
        self.args=args    
        self.VALID_LUCC_INFO=valid_lucc_info 
        self.LUCC_FOLDER = args.lucc_data_path       
        self.MASK_DATA_FILE = args.mask_area       
        self.OUTPUT_FOLDER = args.output_path
        self.LOG_FILE=os.path.join(self.OUTPUT_FOLDER,"log.txt")
        self.SAMPLE_FOLDER=os.path.join(self.OUTPUT_FOLDER,"Samples")
        self.MASK_FOLDER=os.path.join(self.OUTPUT_FOLDER,"Mask")    
        self.BASELINE_FOLDER=os.path.join(self.OUTPUT_FOLDER,"Baseline")   
        self.PARAMETER_FILE=os.path.join(self.OUTPUT_FOLDER,"parameters.json")    
        self.SIMULATION_ROOT=os.path.join(self.OUTPUT_FOLDER,"Simulation") 
        self.PREDICTION_FOLDER=os.path.join(self.OUTPUT_FOLDER,"Prediction")   
        self.Treatment=''
        self.SIMULATION_FOLDER=''      
        self.ACCURACY_FOLDER=''
        self.TEST_FOLDER=''        
        self.MODEL_FILE=''     
        self.ACCURACY_FILE=''            
        
        self.__folders=[self.OUTPUT_FOLDER,self.SAMPLE_FOLDER, self.MASK_FOLDER,
                        self.BASELINE_FOLDER,self.SIMULATION_ROOT]        
    
    def check_folder(self, directory_path):
        result = {'status': 'success', 'message': ''}
        try:
            if not os.path.exists(directory_path):
                os.mkdir(directory_path)  
            else:
                shutil.rmtree(directory_path)
                os.mkdir(directory_path) 
        except Exception as e:
            result['status']='error'
            result['message']=f"Error: {e}"        
        return result
    
    def check_treatment_folders(self,TreatmentID):
        if TreatmentID!='':
            TreatmentID=str(TreatmentID)
            self.SIMULATION_FOLDER=os.path.join(self.SIMULATION_ROOT,"R"+TreatmentID)        
            self.ACCURACY_FOLDER=os.path.join(self.SIMULATION_FOLDER,"Accuracy")
            self.TEST_FOLDER=os.path.join(self.SIMULATION_FOLDER,"Test")               
            self.MODEL_FILE=os.path.join(self.SIMULATION_FOLDER,"trained_model.pth")        
            self.ACCURACY_FILE=os.path.join(self.SIMULATION_FOLDER,"accuracy.shp")
            self.check_folder(self.SIMULATION_FOLDER)
            self.check_folder(self.ACCURACY_FOLDER)
            self.check_folder(self.TEST_FOLDER)
            self.check_folder(self.PREDICTION_FOLDER)


    def check_folders(self):
        result = {'status': 'success', 'message': ''}
        for folder in self.__folders:
            cur_result = self.check_folder(folder)
            if cur_result['status'] != 'success':
                result['status'] = 'error'
                result['message']= cur_result['message']
                return result        
        return result
    
    def clear_temporary_files(self):
        result = {'status': 'success', 'message': ''}
        shutil.rmtree(self.BASELINE_FOLDER)
        shutil.rmtree(self.MASK_FOLDER)
        shutil.rmtree(self.SAMPLE_FOLDER)              
        return result
        


