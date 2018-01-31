package hex.ensemble;

import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;

import hex.Model;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.StackedEnsembleModel;
import hex.StackedEnsembleModel.StackedEnsembleParameters;
import hex.deeplearning.DeepLearning;
import hex.deeplearning.DeepLearningModel;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.schemas.DRFV3;
import hex.schemas.DeepLearningV3;
import hex.schemas.GBMV3;
import hex.schemas.GLMV3;
import hex.tree.drf.DRF;
import hex.tree.drf.DRFModel;
import hex.tree.gbm.GBM;
import hex.tree.gbm.GBMModel;

import water.DKV;
import water.Job;
import water.Key;
import water.exceptions.H2OIllegalArgumentException;
import water.fvec.Frame;
import water.util.Log;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

class Metalearner {

    private Frame _levelOneTrainingFrame;
    private Frame _levelOneValidationFrame;
    private String _metalearner_params;
    private StackedEnsembleModel _model;
    private Job _job;
    private Key<Model> _metalearnerKey;
    private Job _metalearnerJob;
    private StackedEnsembleParameters _parms;
    private boolean _hasMetalearnerParams;
    
    Metalearner(Frame levelOneTrainingFrame, Frame levelOneValidationFrame, String metalearner_params,
                       StackedEnsembleModel model, Job StackedEnsembleJob, Key<Model> metalearnerKey, Job metalearnerJob,
                       StackedEnsembleParameters parms, boolean hasMetalearnerParams){

        _levelOneTrainingFrame = levelOneTrainingFrame;
        _levelOneValidationFrame = levelOneValidationFrame;
        _metalearner_params = metalearner_params;
        _model = model;
        _job = StackedEnsembleJob;
        _metalearnerKey = metalearnerKey;
        _metalearnerJob = metalearnerJob;
        _parms = parms;
        _hasMetalearnerParams = hasMetalearnerParams;

    }

    void computeAutoMetalearner(){
        //GLM Metalearner
        GLM metaGLMBuilder = ModelBuilder.make("GLM", _metalearnerJob, _metalearnerKey);

        metaGLMBuilder._parms._non_negative = true;
        //metaGLMBuilder._parms._alpha = new double[] {0.0, 0.25, 0.5, 0.75, 1.0};
        metaGLMBuilder._parms._train = _levelOneTrainingFrame._key;
        metaGLMBuilder._parms._valid = (_levelOneValidationFrame == null ? null : _levelOneValidationFrame._key);
        metaGLMBuilder._parms._response_column = _model.responseColumn;
        if (_model._parms._metalearner_fold_column == null) {
            metaGLMBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
            if (_model._parms._metalearner_nfolds > 1) {
                if (_model._parms._metalearner_fold_assignment == null) {
                    metaGLMBuilder._parms._fold_assignment = Model.Parameters.FoldAssignmentScheme.AUTO;
                } else {
                    metaGLMBuilder._parms._fold_assignment = _model._parms._metalearner_fold_assignment;  //cross-validation of the metalearner
                }
            }
        } else {
            metaGLMBuilder._parms._fold_column = _model._parms._metalearner_fold_column;  //cross-validation of the metalearner
        }

        // Enable lambda search if a validation frame is passed in to get a better GLM fit.
        // Since we are also using non_negative to true, we should also set early_stopping = false.
        if (metaGLMBuilder._parms._valid != null) {
            metaGLMBuilder._parms._lambda_search = true;
            metaGLMBuilder._parms._early_stopping = false;
        }
        if (_model.modelCategory == ModelCategory.Regression) {
            metaGLMBuilder._parms._family = GLMModel.GLMParameters.Family.gaussian;
        } else if (_model.modelCategory == ModelCategory.Binomial) {
            metaGLMBuilder._parms._family = GLMModel.GLMParameters.Family.binomial;
        } else if (_model.modelCategory == ModelCategory.Multinomial) {
            metaGLMBuilder._parms._family = GLMModel.GLMParameters.Family.multinomial;
        } else {
            throw new H2OIllegalArgumentException("Family " + _model.modelCategory + "  is not supported.");
        }

        metaGLMBuilder.init(false);

        Job<GLMModel> j = metaGLMBuilder.trainModel();

        while (j.isRunning()) {
            try {
                _job.update(j._work, "training metalearner(" + _model._parms._metalearner_algorithm + ")");
                Thread.sleep(100);
            } catch (InterruptedException e) {
            }
        }

        Log.info("Finished training metalearner model(" + _model._parms._metalearner_algorithm + ").");

        _model._output._metalearner = metaGLMBuilder.get();
        _model.doScoreOrCopyMetrics(_job);
        if (_parms._keep_levelone_frame) {
            _model._output._levelone_frame_id = _levelOneTrainingFrame; //Keep Level One Training Frame in Stacked Ensemble model object
        } else {
            DKV.remove(_levelOneTrainingFrame._key); //Remove Level One Training Frame from DKV
        }
        if (null != _levelOneValidationFrame) {
            DKV.remove(_levelOneValidationFrame._key); //Remove Level One Validation Frame from DKV
        }
        _model.update(_job);
        _model.unlock(_job);
    }

    void computeGBMMetalearner(){
        //GBM Metalearner
        GBM metaGBMBuilder;
        metaGBMBuilder = ModelBuilder.make("GBM", _metalearnerJob, _metalearnerKey);
        GBMV3.GBMParametersV3 params = new GBMV3.GBMParametersV3();
        params.init_meta();
        params.fillFromImpl(metaGBMBuilder._parms); // Defaults for this builder into schema

        //Metalearner parameters
        if (_hasMetalearnerParams) {
            Properties p = new Properties();
            HashMap<String, String[]> map = new Gson().fromJson(_metalearner_params, new TypeToken<HashMap<String, String[]>>() {
            }.getType());
            for (Map.Entry<String, String[]> param : map.entrySet()) {
                String[] paramVal = param.getValue();
                if (paramVal.length == 1) {
                    p.setProperty(param.getKey(), paramVal[0]);
                } else {
                    p.setProperty(param.getKey(), Arrays.toString(paramVal));
                }
                params.fillFromParms(p, true);
            }
            GBMModel.GBMParameters gbmParams = params.createAndFillImpl();
            metaGBMBuilder._parms = gbmParams;
        }

        metaGBMBuilder._parms._train = _levelOneTrainingFrame._key;
        metaGBMBuilder._parms._valid = (_levelOneValidationFrame == null ? null : _levelOneValidationFrame._key);
        metaGBMBuilder._parms._response_column = _model.responseColumn;
        metaGBMBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
        if (_model._parms._metalearner_fold_column == null) {
            metaGBMBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
            if (_model._parms._metalearner_nfolds > 1) {
                if (_model._parms._metalearner_fold_assignment == null) {
                    metaGBMBuilder._parms._fold_assignment = Model.Parameters.FoldAssignmentScheme.AUTO;
                } else {
                    metaGBMBuilder._parms._fold_assignment = _model._parms._metalearner_fold_assignment;  //cross-validation of the metalearner
                }
            }
        } else {
            metaGBMBuilder._parms._fold_column = _model._parms._metalearner_fold_column;  //cross-validation of the metalearner
        }

        metaGBMBuilder.init(false);

        Job<GBMModel> j = metaGBMBuilder.trainModel();

        while (j.isRunning()) {
            try {
                _job.update(j._work, "training metalearner(" + _model._parms._metalearner_algorithm + ")");
                Thread.sleep(100);
            } catch (InterruptedException e) {
            }
        }

        Log.info("Finished training metalearner model(" + _model._parms._metalearner_algorithm + ").");

        _model._output._metalearner = metaGBMBuilder.get();
        _model.doScoreOrCopyMetrics(_job);
        if (_parms._keep_levelone_frame) {
            _model._output._levelone_frame_id = _levelOneTrainingFrame; //Keep Level One Training Frame in Stacked Ensemble model object
        } else {
            DKV.remove(_levelOneTrainingFrame._key); //Remove Level One Training Frame from DKV
        }
        if (null != _levelOneValidationFrame) {
            DKV.remove(_levelOneValidationFrame._key); //Remove Level One Validation Frame from DKV
        }
        _model.update(_job);
        _model.unlock(_job);

    }

    void computeDRFMetalearner(){
        //DRF Metalearner
        DRF metaDRFBuilder;
        metaDRFBuilder = ModelBuilder.make("DRF", _metalearnerJob, _metalearnerKey);
        DRFV3.DRFParametersV3 params = new DRFV3.DRFParametersV3();
        params.init_meta();
        params.fillFromImpl(metaDRFBuilder._parms); // Defaults for this builder into schema

        //Metalearner parameters
        if (_hasMetalearnerParams) {
            Properties p = new Properties();
            HashMap<String, String[]> map = new Gson().fromJson(_metalearner_params, new TypeToken<HashMap<String, String[]>>() {
            }.getType());
            for (Map.Entry<String, String[]> param : map.entrySet()) {
                String[] paramVal = param.getValue();
                if (paramVal.length == 1) {
                    p.setProperty(param.getKey(), paramVal[0]);
                } else {
                    p.setProperty(param.getKey(), Arrays.toString(paramVal));
                }
                params.fillFromParms(p, true);
            }
            DRFModel.DRFParameters drfParams = params.createAndFillImpl();
            metaDRFBuilder._parms = drfParams;
        }
        metaDRFBuilder._parms._train = _levelOneTrainingFrame._key;
        metaDRFBuilder._parms._valid = (_levelOneValidationFrame == null ? null : _levelOneValidationFrame._key);
        metaDRFBuilder._parms._response_column = _model.responseColumn;
        metaDRFBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
        if (_model._parms._metalearner_fold_column == null) {
            metaDRFBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
            if (_model._parms._metalearner_nfolds > 1) {
                if (_model._parms._metalearner_fold_assignment == null) {
                    metaDRFBuilder._parms._fold_assignment = Model.Parameters.FoldAssignmentScheme.AUTO;
                } else {
                    metaDRFBuilder._parms._fold_assignment = _model._parms._metalearner_fold_assignment;  //cross-validation of the metalearner
                }
            }
        } else {
            metaDRFBuilder._parms._fold_column = _model._parms._metalearner_fold_column;  //cross-validation of the metalearner
        }

        metaDRFBuilder.init(false);

        Job<DRFModel> j = metaDRFBuilder.trainModel();

        while (j.isRunning()) {
            try {
                _job.update(j._work, "training metalearner(" + _model._parms._metalearner_algorithm + ")");
                Thread.sleep(100);
            } catch (InterruptedException e) {
            }
        }

        Log.info("Finished training metalearner model(" + _model._parms._metalearner_algorithm + ").");

        _model._output._metalearner = metaDRFBuilder.get();
        _model.doScoreOrCopyMetrics(_job);
        if (_parms._keep_levelone_frame) {
            _model._output._levelone_frame_id = _levelOneTrainingFrame; //Keep Level One Training Frame in Stacked Ensemble model object
        } else {
            DKV.remove(_levelOneTrainingFrame._key); //Remove Level One Training Frame from DKV
        }
        if (null != _levelOneValidationFrame) {
            DKV.remove(_levelOneValidationFrame._key); //Remove Level One Validation Frame from DKV
        }
        _model.update(_job);
        _model.unlock(_job);
    }

    void computeGLMMetalearner(){
        //GLM Metalearner
        GLM metaGLMBuilder;
        metaGLMBuilder = ModelBuilder.make("GLM", _metalearnerJob, _metalearnerKey);
        GLMV3.GLMParametersV3 params = new GLMV3.GLMParametersV3();
        params.init_meta();
        params.fillFromImpl(metaGLMBuilder._parms); // Defaults for this builder into schema

        //Metalearner parameters
        if (_hasMetalearnerParams) {
            Properties p = new Properties();
            HashMap<String, String[]> map = new Gson().fromJson(_metalearner_params, new TypeToken<HashMap<String, String[]>>() {
            }.getType());
            for (Map.Entry<String, String[]> param : map.entrySet()) {
                String[] paramVal = param.getValue();
                if (paramVal.length == 1) {
                    p.setProperty(param.getKey(), paramVal[0]);
                } else {
                    p.setProperty(param.getKey(), Arrays.toString(paramVal));
                }
                params.fillFromParms(p, true);
            }
            GLMModel.GLMParameters glmParams = params.createAndFillImpl();
            metaGLMBuilder._parms = glmParams;
        }

        metaGLMBuilder._parms._train = _levelOneTrainingFrame._key;
        metaGLMBuilder._parms._valid = (_levelOneValidationFrame == null ? null : _levelOneValidationFrame._key);
        metaGLMBuilder._parms._response_column = _model.responseColumn;
        metaGLMBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
        if (_model._parms._metalearner_fold_column == null) {
            metaGLMBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
            if (_model._parms._metalearner_nfolds > 1) {
                if (_model._parms._metalearner_fold_assignment == null) {
                    metaGLMBuilder._parms._fold_assignment = Model.Parameters.FoldAssignmentScheme.AUTO;
                } else {
                    metaGLMBuilder._parms._fold_assignment = _model._parms._metalearner_fold_assignment;  //cross-validation of the metalearner
                }
            }
        } else {
            metaGLMBuilder._parms._fold_column = _model._parms._metalearner_fold_column;  //cross-validation of the metalearner
        }

        if (_model.modelCategory == ModelCategory.Regression) {
            metaGLMBuilder._parms._family = GLMModel.GLMParameters.Family.gaussian;
        } else if (_model.modelCategory == ModelCategory.Binomial) {
            metaGLMBuilder._parms._family = GLMModel.GLMParameters.Family.binomial;
        } else if (_model.modelCategory == ModelCategory.Multinomial) {
            metaGLMBuilder._parms._family = GLMModel.GLMParameters.Family.multinomial;
        } else {
            throw new H2OIllegalArgumentException("Family " + _model.modelCategory + "  is not supported.");
        }
        metaGLMBuilder.init(false);

        Job<GLMModel> j = metaGLMBuilder.trainModel();

        while (j.isRunning()) {
            try {
                _job.update(j._work, "training metalearner(" + _model._parms._metalearner_algorithm + ")");
                Thread.sleep(100);
            } catch (InterruptedException e) {
            }
        }

        Log.info("Finished training metalearner model(" + _model._parms._metalearner_algorithm + ").");

        _model._output._metalearner = metaGLMBuilder.get();
        _model.doScoreOrCopyMetrics(_job);
        if (_parms._keep_levelone_frame) {
            _model._output._levelone_frame_id = _levelOneTrainingFrame; //Keep Level One Training Frame in Stacked Ensemble model object
        } else {
            DKV.remove(_levelOneTrainingFrame._key); //Remove Level One Training Frame from DKV
        }
        if (null != _levelOneValidationFrame) {
            DKV.remove(_levelOneValidationFrame._key); //Remove Level One Validation Frame from DKV
        }
        _model.update(_job);
        _model.unlock(_job);

    }

    void computeDeepLearningMetalearner(){
        //DeepLearning Metalearner
        DeepLearning metaDeepLearningBuilder;
        metaDeepLearningBuilder = ModelBuilder.make("DeepLearning", _metalearnerJob, _metalearnerKey);
        DeepLearningV3.DeepLearningParametersV3 params = new DeepLearningV3.DeepLearningParametersV3();
        params.init_meta();
        params.fillFromImpl(metaDeepLearningBuilder._parms); // Defaults for this builder into schema

        //Metalearner parameters
        if (_hasMetalearnerParams) {
            Properties p = new Properties();
            HashMap<String, String[]> map = new Gson().fromJson(_metalearner_params, new TypeToken<HashMap<String, String[]>>() {
            }.getType());
            for (Map.Entry<String, String[]> param : map.entrySet()) {
                String[] paramVal = param.getValue();
                if (paramVal.length == 1) {
                    p.setProperty(param.getKey(), paramVal[0]);
                } else {
                    p.setProperty(param.getKey(), Arrays.toString(paramVal));
                }
                params.fillFromParms(p, true);
            }
            DeepLearningModel.DeepLearningParameters dlParams = params.createAndFillImpl();
            metaDeepLearningBuilder._parms = dlParams;
        }
        metaDeepLearningBuilder._parms._train = _levelOneTrainingFrame._key;
        metaDeepLearningBuilder._parms._valid = (_levelOneValidationFrame == null ? null : _levelOneValidationFrame._key);
        metaDeepLearningBuilder._parms._response_column = _model.responseColumn;
        metaDeepLearningBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
        if (_model._parms._metalearner_fold_column == null) {
            metaDeepLearningBuilder._parms._nfolds = _model._parms._metalearner_nfolds;  //cross-validation of the metalearner
            if (_model._parms._metalearner_nfolds > 1) {
                if (_model._parms._metalearner_fold_assignment == null) {
                    metaDeepLearningBuilder._parms._fold_assignment = Model.Parameters.FoldAssignmentScheme.AUTO;
                } else {
                    metaDeepLearningBuilder._parms._fold_assignment = _model._parms._metalearner_fold_assignment;  //cross-validation of the metalearner
                }
            }
        } else {
            metaDeepLearningBuilder._parms._fold_column = _model._parms._metalearner_fold_column;  //cross-validation of the metalearner
        }

        metaDeepLearningBuilder.init(false);

        Job<DeepLearningModel> j = metaDeepLearningBuilder.trainModel();

        while (j.isRunning()) {
            try {
                _job.update(j._work, "training metalearner(" + _model._parms._metalearner_algorithm + ")");
                Thread.sleep(100);
            } catch (InterruptedException e) {
            }
        }

        Log.info("Finished training metalearner model(" + _model._parms._metalearner_algorithm + ").");

        _model._output._metalearner = metaDeepLearningBuilder.get();
        _model.doScoreOrCopyMetrics(_job);
        if (_parms._keep_levelone_frame) {
            _model._output._levelone_frame_id = _levelOneTrainingFrame; //Keep Level One Training Frame in Stacked Ensemble model object
        } else {
            DKV.remove(_levelOneTrainingFrame._key); //Remove Level One Training Frame from DKV
        }
        if (null != _levelOneValidationFrame) {
            DKV.remove(_levelOneValidationFrame._key); //Remove Level One Validation Frame from DKV
        }
        _model.update(_job);
        _model.unlock(_job);
    }
}
