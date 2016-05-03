package core;

import helpers.Checkers;
import helpers.DataLoader;
import helpers.DirUtils;
import helpers.ParamLoader;
import helpers.ParamSaver;
import helpers.UtilFuncs;

import java.io.IOException;
import java.util.Optional;

import myUtil.Savers;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Errors;
import defs.Hypers;
import defs.InvalidModelException;
import defs.Model;
import defs.NonConvergeException;
import defs.ParamModelMismatchException;
import defs.Params;
import defs.SoRecParams;
import defs.SocBIT_Params;

public class Experiment {
	
	static Dataset train_ds;
	private static RealMatrix test_ratings;
	private static Params gt_params;	// only exist in synthetic data
	private static int gt_numTopic;
	
	public static void main(String[] args) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		int runIndex = 3;
		int numUser = 2000;
//		synExp(numUser, runIndex);
		
		realExp();
	}

	private static void synExp(int numUser, int runIndex) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		String parentDir = "data/syn/N" + numUser + "/";
		String splitDir = parentDir   ; // + "/1_split/"  
		String graphDir = parentDir;
		String resDir = "result/syn/N" + numUser + "/run" + runIndex + "/" ;
		// turn this on if run on synthetic data
		loadDataSets(splitDir, graphDir); 	// for synthetic data, graphDir and dataDir are the same
		getGroundTruth(parentDir);
		runSynExpAndSave(resDir);
	}

	private static void runSynExpAndSave(String resDir) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		// TODO Auto-generated method stub
		String errDir = resDir + "errors/"; 		DirUtils.mkDir(errDir);
		String trainErrStr = mkTitle4ErrStr();
		String testErrs = "testErr_soRec, testErr_socBIT \n";
		String paramErr = "";
		
//		int minK = 2; int maxK = 10;
//		int minK = gt_numTopic; int maxK = gt_numTopic;	// for fast testing
//		for (int numTopic = minK; numTopic <=  maxK; numTopic++) 
		int[] Ks = {5, 10};
		for (int numTopic : Ks)
		{
			Model socBIT = trainBySocBIT(train_ds, numTopic);
			Model soRec = trainBySoRec(train_ds, numTopic);
			
			trainErrStr += numTopic + "," + soRec.toErrString() + "," + socBIT.toErrString() + "\n";
			
			saveLearnedParams(soRec, socBIT, numTopic, resDir);
			//XXX: tmp turn off prediction as we are running on whole ds
			double test_rmse_soRec = predict(soRec, test_ratings);
			double test_rmse_socBIT = predict(socBIT, test_ratings);
			testErrs += test_rmse_soRec + "," + test_rmse_socBIT + "\n";
			
			// turn this on if run on synthetic data
			if (numTopic == gt_numTopic) {
				paramErr += getParamErr((SocBIT_Params) socBIT.learnedParams, (SoRecParams) soRec.learnedParams, gt_params);
			}
			
		}	
		
		String fErrors = errDir + "all_errors.csv";
		Savers.save(trainErrStr, fErrors);
		
		String fTest = errDir + "test_err.csv";
		Savers.save(testErrs, fTest);
		
		String fParamErr = errDir + "param_learn_err.csv" ;
		Savers.save(paramErr, fParamErr);
	}

	private static void realExp() throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		String dataPath = "data/real/fq/";
		//		if run on server 
		// dataPath = "/StorageArray2/minhduc/data/fq/sg/";
		
		String graphDir = dataPath;
		
		String[] lower = { "1", "11", "21", "81"};	// "41",  "161"  
		String[] upper = {"10", "20", "40", "160"}; // "80",  "160plus"
		
		for (int i = upper.length - 1; i >= 0; i--) {
			String ub = upper[i];
			String lb = lower[i];
			System.out.println("Run exp on the group of users with numChkin in range [" + lb + "," + ub + "]");
			String dataDir = dataPath + "ds" + ub + "/";
//			String dataDir = path + "ds160plus/"; 
			loadDataSets(dataDir, graphDir);
			String resDir = "result/real/fq/max" + ub + "chkins/";
//			String resDir = "result/real/fq/ds160plus/";
			runRealExpAndSave(resDir);
		}
	}

	private static void runRealExpAndSave(String resDir) throws IOException, InvalidModelException,
														ParamModelMismatchException, NonConvergeException {
		
		String errDir = resDir + "errors/"; 		DirUtils.mkDir(errDir);
		String trainErrStr = mkTitle4ErrStr();
		String testErrs = "testErr_soRec, testErr_socBIT \n";
		//		int minK = 2; int maxK = 10;
//		for (int numTopic = minK; numTopic <=  maxK; numTopic++) 
		int[] Ks = {5, 10};
		for (int numTopic : Ks)
		{
			Model socBIT = trainBySocBIT(train_ds, numTopic);
			Model soRec = trainBySoRec(train_ds, numTopic);
			
			trainErrStr += numTopic + "," + soRec.toErrString() + "," + socBIT.toErrString() + "\n";
			
			saveLearnedParams(soRec, socBIT, numTopic, resDir);
			double test_rmse_soRec = predict(soRec, test_ratings);
			double test_rmse_socBIT = predict(socBIT, test_ratings);
			testErrs += test_rmse_soRec + "," + test_rmse_socBIT + "\n";
		}	
		
		String fErrors = errDir + "all_errors.csv";
		Savers.save(trainErrStr, fErrors);
		
		String fTest = errDir + "test_err.csv";
		Savers.save(testErrs, fTest);
	}

	private static void getGroundTruth(String parentDir) throws IOException {
		String gtParamDir = parentDir + "true_params/";
		gt_numTopic = 5;
		gt_params = ParamLoader.load(gtParamDir);
	}

	private static void loadDataSets(String dataDir, String graphDir) throws IOException {
		@SuppressWarnings("unused")
		int splitIndex = 1;
		
		DataLoader loader = new DataLoader(dataDir);
		String train_rating_file = dataDir +  "train_ratings.csv";	// splitIndex + "_split/" +		 
		String graph_file = graphDir + "edge_weights.csv";
		train_ds = loader.load(train_rating_file, graph_file);	// splitIndex
		System.out.println("Loaded train ds = (ratings and the graph)");
		
		// XXX: tmp turn off for synExp, as synExp currently run on whole ds
		String test_file = dataDir +  "test_ratings.csv";	// splitIndex + "_split/" +
		test_ratings = loader.loadRatings(test_file);
		System.out.println("Loaded test ratings");
		
	}

	private static Trainer initTrainer(String model, Dataset ds, int numTopic) throws InvalidModelException {
		
		int maxIter = 1; // 10
		double topicLambda = 0.001;
		double weightLambda = 1;
		
		Hypers hypers = null;
		if (Checkers.isValid(model)) {
			
			if (model.equalsIgnoreCase("soRec")) {
				hypers = Hypers.setBySoRec(topicLambda, weightLambda);
				System.out.println("Try " + numTopic + " topics");
			}
			
			if (model.equalsIgnoreCase("socBIT")) {
				double brandLambda = 0.001;
				double decisionLambda = 1;
				hypers = Hypers.setBySocBIT(topicLambda, brandLambda, weightLambda, decisionLambda);
				System.out.println("Try " + numTopic + " topics.");
//				printRegConst(hypers);
			}
			
//			if (model.equalsIgnoreCase("STE")) {
//				hypers = Hypers.assignBySTE();
//				System.out.println("Try " + numTopic + " topics.");
//			} 
//			
//			if (model.equalsIgnoreCase("bSTE")) {
//				hypers = Hypers.assignByBSTE();
//				System.out.println("Try " + numTopic + " topics.");
//			}
		}
		
		else {
			throw new InvalidModelException();
		}
		
		Trainer trainer = new Trainer(model, ds, numTopic, hypers, maxIter);
		return trainer;
	}

	private static Model trainBySoRec(Dataset ds, int numTopic) throws InvalidModelException, IOException, ParamModelMismatchException, NonConvergeException {
		
		System.out.println("Training by soRec model");
		Trainer trainer = initTrainer("soRec", ds, numTopic);
		SoRecParams initParams = new SoRecParams(ds.numUser, ds.numItem, numTopic);
		Model result = trainer.trainByGD(initParams);
		
		return result;
	}
	
	private static Model trainBySocBIT(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		System.out.println("Training by socBIT model");
		
		Trainer trainer = initTrainer("socBIT", ds, numTopic);	// currently training on whole data set, switch to training set later	
		SocBIT_Params initParams = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, trainer.numTopic);
		System.out.println("iter, obj_value (rating + regs + edge_weight_errors), rating errors");
		Model result = trainer.trainByGD(initParams);
		return result;
	}
	
	/**
	 * Make predictions using specified {@link params} and check with ground truth {@link test_ds} to obtain ... 
	 * evaluation metrics
	 * @param params
	 * @param test_ds
	 * @return 
	 */
	private static double predict(Model model, RealMatrix test_ratings) {
		
		double rmse = 0;
		
		RealMatrix errMat = null;
		RecSysCal calculator = model.calculator;
		if (calculator instanceof SoRec_Cal) {
			System.out.println("Predict ratings by SoRec...");
			SoRec_Cal soRec_Cal = (SoRec_Cal) calculator;
			RealMatrix estRatings = soRec_Cal.estRatings(model.learnedParams);
			errMat = soRec_Cal.calRatingErrors(estRatings, test_ratings);
		}
		if (calculator instanceof SocBIT_Cal) {
			System.out.println("Predict ratings by SocBIT...");
			SocBIT_Cal socBIT_Cal = (SocBIT_Cal) calculator;
			RealMatrix estRatings = socBIT_Cal.estRatings(model.learnedParams);
			errMat = socBIT_Cal.calRatingErrors(estRatings, test_ratings);
		}
		
		int na_marker = -1;
		int numRating = UtilFuncs.numNeq(test_ratings, na_marker);	
		rmse = calRMSE(errMat, numRating);
		return rmse;
	}

	private static double calRMSE(RealMatrix errMat, int numErr) {
		double rmse;
		double mse = UtilFuncs.sqFrobNorm(errMat)/numErr;
		rmse = Math.sqrt(mse);
		return rmse;
	}
	
	private static String concat(String model, Errors errors) {
		return model + "," + 	errors.toString();
	}

	private static Errors compDiff(Params params1, Params params2) {
		
		Errors errors = null;
		if ((params1 instanceof SocBIT_Params) && (params2 instanceof SocBIT_Params)) {
			
			SocBIT_Params cast_p1 = (SocBIT_Params) params1;
			SocBIT_Params cast_p2 = (SocBIT_Params) params2;
			double topicUserErr = cast_p1.topicUser.subtract(cast_p2.topicUser).getFrobeniusNorm();
			double topicItemErr = cast_p1.topicItem.subtract(cast_p2.topicItem).getFrobeniusNorm();
			
			Double brandUserErr = cast_p1.brandUser.subtract(cast_p2.brandUser).getFrobeniusNorm();
			Double brandItemErr = cast_p1.brandItem.subtract(cast_p2.brandItem).getFrobeniusNorm();
			Double decisionPrefErr = toVector(cast_p1.userDecisionPrefs).subtract(toVector(cast_p2.userDecisionPrefs)).getNorm();
			errors = new Errors(topicUserErr, topicItemErr, Optional.of(brandUserErr), Optional.of(brandItemErr), Optional.of(decisionPrefErr));
		} 
		else {
			double topicUserErr = params1.topicUser.subtract(params2.topicUser).getFrobeniusNorm();
			double topicItemErr = params1.topicItem.subtract(params2.topicItem).getFrobeniusNorm();
			Optional<Double> empty = Optional.empty();
			errors = new Errors(topicUserErr, topicItemErr, empty, empty, empty);
		}
		return errors;
	}

	@SuppressWarnings("unused")
	private static void printRegConst(Hypers hypers) {
		System.out.println("Regularization constants: ");
		System.out.println("topicLambda, brandLambda, weightLambda, decisionLambda" );
		System.out.println(hypers.topicLambda + "," + hypers.brandLambda + "," + hypers.weightLambda + "," + hypers.decisionLambda);
	}

	private static RealVector toVector(double[] arr) {
		return new ArrayRealVector(arr);
	}
	
	private static String mkTitle4ErrStr() {
		String allErrStr = "numTopic, "	+ "ratingErr_soRec, trustErr_soRec, objValue_soRec, "
										+ "ratingErr_socBIT,  trustErr_socBIT, objValue_socBIT "; 
										
		allErrStr += "\n";
		return allErrStr;
	}

	private static void saveLearnedParams(Model soRec_result, Model socBIT_result, int numTopic, String resDir)
			throws IOException {
		
		String model = "soRec";
		SoRecParams soRecParams = (SoRecParams) soRec_result.learnedParams;
		save(soRecParams, model, numTopic, resDir);
		
		model = "socBIT";
		SocBIT_Params socBIT_params = (SocBIT_Params) socBIT_result.learnedParams;
		save(socBIT_params, model, numTopic, resDir);
		
//			model = "STE";
//			Params ste_params = ste_result.learnedParams;
//			save(ste_params, model, numTopic, resDir);
//			
//			model = "bSTE";
//			SocBIT_Params bSTE_params = (SocBIT_Params) bSTE_res.learnedParams;
//			save(bSTE_params, model, numTopic, resDir);
	}
	
	private static void save(Params params, String model, int numTopic, String resDir) throws IOException {
		
		String name = resDir + model + "/" + "numTopic" + numTopic + "/";
		DirUtils.mkDir(name);
		ParamSaver.save(params, name);
	}

	@SuppressWarnings("unused")
	private static Model trainByBSTE(Dataset ds, int numTopic) throws InvalidModelException, IOException, ParamModelMismatchException, NonConvergeException {
		
		System.out.println("Training by bSTE model...");
		Trainer trainer = initTrainer("bSTE", ds, numTopic);
		
		SocBIT_Params initParams = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, trainer.numTopic);
		System.out.println("iter, obj_value (rating + regs), rating errors");
		Model result = trainer.trainByGD(initParams);
		return result;
	}
	
	@SuppressWarnings("unused")
	private static Model trainBySTE(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		System.out.println("Training by STE model...");
		
		Trainer trainer = initTrainer("STE", ds, numTopic);	// currently training on whole data set, switch to training set later	
		Params initParams = new Params(ds.numUser, ds.numItem, trainer.numTopic);
		initParams.createFeatsUniformly();
		System.out.println("iter, obj_value (rating + regs), rating errors");
		Model result = trainer.trainByGD(initParams);
		
		return result;
	}
	
	private static String getParamErr(SocBIT_Params socBIT_params, SoRecParams soRecParams, Params gt_params) {
		
		String paramErr = "model, topicUserErr, topicItemErr, brandUserErr, brandItemErr, decisionPrefErr \n";
		Errors socBIT_errors = compDiff( socBIT_params, (SocBIT_Params) gt_params);
		Errors soRecErrs = compDiff(soRecParams, gt_params);
		
		paramErr += concat("socBIT",  socBIT_errors) + "\n";
		paramErr += concat("soRec", soRecErrs) + "\n";
			
		return paramErr;
	}
}
