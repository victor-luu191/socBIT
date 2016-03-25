package core;

import helpers.Checkers;
import helpers.DataLoader;
import helpers.DirUtils;
import helpers.ParamLoader;
import helpers.ParamSaver;
import helpers.UtilFuncs;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Optional;

import myUtil.Savers;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Errors;
import defs.Hypers;
import defs.InvalidModelException;
import defs.NonConvergeException;
import defs.ParamModelMismatchException;
import defs.Params;
import defs.Model;
import defs.SoRecParams;
import defs.SocBIT_Params;

public class Experiment {
	
	static Dataset train_ds;
	static Dataset test_ds;
	
	public static void main(String[] args) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		// temporary value passing, later will read from file data_stats or itemInfo
		int numUser = 2000;	// 
		int gt_numTopic = 5;
		int numBrand = 9*gt_numTopic + 1;	

		// TODO: switch to the pkg org.kohsuke.args4j to enable named args
		String dataDir = "data/syn/N" + numUser + "/unif/";	//  or args[0]
		int splitIndex = 1;
		String train_file = dataDir + splitIndex + "_split/" + "train_ratings.csv";
		Dataset train_ds = DataLoader.load(dataDir, numBrand, splitIndex, train_file);
		String test_file = dataDir + splitIndex + "_split/" + "test_ratings.csv";
		test_ds = DataLoader.load(dataDir, numBrand, splitIndex, test_file);
		
		String gtParamDir = dataDir + "true_params/";
		Params gt_params = ParamLoader.load(gtParamDir);
		
		String resDir = "result/syn/N" + train_ds.numUser + "/";
		String errDir = resDir + "errors/"; 		DirUtils.mkDir(errDir);
		
		String allErrStr = mkTitle4ErrStr();
		
		int minK = 2; int maxK = 10;
//		int minK = gt_numTopic; int maxK = gt_numTopic;	// for fast testing
		for (int numTopic = minK; numTopic <=  maxK; numTopic++) {
			
			Model soRec = trainBySoRec(train_ds, numTopic);
			Model socBIT = trainBySocBIT(train_ds, numTopic);
			allErrStr += numTopic + "," + soRec.toErrString() + "," + socBIT.toErrString() + "\n";
			
			saveLearnedParams(soRec, socBIT, numTopic, resDir);

			if (numTopic == gt_numTopic) {
				double test_rmse_soRec = predict(soRec, test_ds);
				double test_rmse_socBIT = predict(socBIT, test_ds);
				
//				String paramErr = getParamErr(socBIT_params, ste_params, bSTE_params, gt_params);
//				String fParamErr = errDir + "param_learn_err.csv" ;
//				Savers.save(paramErr, fParamErr);
			}
			
//			Result ste_result = trainBySTE(ds, numTopic);
//			Result bSTE_res = trainByBSTE(ds, numTopic);
//			allErrStr += "STE, " + numTopic + "," + ste_result.toErrString() + "\n";
//			allErrStr += "bSTE, " + numTopic + "," + bSTE_res.toErrString() + "\n";
//			
		}	
		
		String fErrors = errDir + "all_errors.csv";
		Savers.save(allErrStr, fErrors);
		
		
//		predict(socBIT_params, test_ds);
	}

	private static Trainer initTrainer(String model, Dataset ds, int numTopic) throws InvalidModelException {
		
		int maxIter = 10;
		double topicLambda = 1;
		double weightLambda = 0.001;
		
		Hypers hypers = null;
		if (Checkers.isValid(model)) {
			
			if (model.equalsIgnoreCase("soRec")) {
				hypers = Hypers.setBySoRec(topicLambda, weightLambda);
				System.out.println("Try " + numTopic + " topics");
			}
			
			if (model.equalsIgnoreCase("socBIT")) {
				double brandLambda = 5;
				double decisionLambda = 2;
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
		Model result = trainer.gradDescent(initParams);
		
		return result;
	}
	
	private static Model trainBySocBIT(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		System.out.println("Training by socBIT model");
		
		Trainer trainer = initTrainer("socBIT", ds, numTopic);	// currently training on whole data set, switch to training set later	
		SocBIT_Params initParams = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, trainer.numTopic);
		System.out.println("iter, obj_value (rating + regs + edge_weight_errors), rating errors");
		Model result = trainer.gradDescent(initParams);
		return result;
	}
	
	/**
	 * Make predictions using specified {@link params} and check with ground truth {@link test_ds} to obtain ... 
	 * evaluation metrics
	 * @param params
	 * @param test_ds
	 * @return 
	 */
	private static double predict(Model model, Dataset test_ds) {
		
		double rmse = 0;
		
		RealMatrix ratings = test_ds.ratings;
		RealMatrix errMat = null;
		RecSysCal calculator = model.calculator;
		if (calculator instanceof SoRec_Cal) {
			SoRec_Cal soRec_Cal = (SoRec_Cal) calculator;
			RealMatrix estRatings = soRec_Cal.estRatings(model.learnedParams);
			errMat = soRec_Cal.calRatingErrors(estRatings, ratings);
		}
		if (calculator instanceof SocBIT_Cal) {
			SocBIT_Cal socBIT_Cal = (SocBIT_Cal) calculator;
			RealMatrix estRatings = socBIT_Cal.estRatings(model.learnedParams);
			errMat = socBIT_Cal.calRatingErrors(estRatings, ratings);
		}
		
		int numRating = count(ratings);
		rmse = calRMSE(errMat, numRating);
		return rmse;
	}

	private static double calRMSE(RealMatrix errMat, int numErr) {
		double rmse;
		double mse = UtilFuncs.sqFrobNorm(errMat)/numErr;
		rmse = Math.sqrt(mse);
		return rmse;
	}
	
	/**
	 * @param matrix
	 * @return number of non-NA entries
	 */
	private static int count(RealMatrix matrix) {
		
		int c = 0;
		for (int i = 0; i < matrix.getRowDimension(); i++) {
			for (int j = 0; j < matrix.getColumnDimension(); j++) {
				if (matrix.getEntry(i, j) != -1) {// non-NA
					c++;
				}
			}
		}
		return c;
	}

	private static String concat(String model, Errors errors) {
		return model + "," + 	errors.toString();
	}

	private static Errors compDiff(Params params1, Params params2) {
		
		if ((params1 instanceof SocBIT_Params) && (params2 instanceof SocBIT_Params)) {
			
			SocBIT_Params cast_p1 = (SocBIT_Params) params1;
			SocBIT_Params cast_p2 = (SocBIT_Params) params2;
			double topicUserErr = cast_p1.topicUser.subtract(cast_p2.topicUser).getFrobeniusNorm();
			double topicItemErr = cast_p1.topicItem.subtract(cast_p2.topicItem).getFrobeniusNorm();
			
			Double brandUserErr = cast_p1.brandUser.subtract(cast_p2.brandUser).getFrobeniusNorm();
			Double brandItemErr = cast_p1.brandItem.subtract(cast_p2.brandItem).getFrobeniusNorm();
			Double decisionPrefErr = toVector(cast_p1.userDecisionPrefs).subtract(toVector(cast_p2.userDecisionPrefs)).getNorm();
			return new Errors(topicUserErr, topicItemErr, Optional.of(brandUserErr), Optional.of(brandItemErr), Optional.of(decisionPrefErr));
		} 
		else {
			double topicUserErr = params1.topicUser.subtract(params2.topicUser).getFrobeniusNorm();
			double topicItemErr = params1.topicItem.subtract(params2.topicItem).getFrobeniusNorm();
			Optional<Double> empty = Optional.empty();
			return new Errors(topicUserErr, topicItemErr, empty, empty, empty);
		}
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
										+ "ratingErr_socBIT,  trustErr_socBIT, objValue_socBIT" ;
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
		Model result = trainer.gradDescent(initParams);
		return result;
	}
	
	@SuppressWarnings("unused")
	private static Model trainBySTE(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		System.out.println("Training by STE model...");
		
		Trainer trainer = initTrainer("STE", ds, numTopic);	// currently training on whole data set, switch to training set later	
		Params initParams = new Params(ds.numUser, ds.numItem, trainer.numTopic);
		initParams.createFeatsUniformly();
		System.out.println("iter, obj_value (rating + regs), rating errors");
		Model result = trainer.gradDescent(initParams);
		
		return result;
	}
	
	@SuppressWarnings("unused")
	private static String getParamErr(SocBIT_Params socBIT_params, Params ste_params, SocBIT_Params bSTE_params, Params gt_params) {
		
		String paramErr = "model, topicUserErr, topicItemErr, brandUserErr, brandItemErr, decisionPrefErr \n";
		Errors socBIT_errors = compDiff( socBIT_params, (SocBIT_Params) gt_params);
		Errors ste_errors = compDiff(ste_params, gt_params);
		Errors bSTE_errors = compDiff(bSTE_params, (SocBIT_Params) gt_params);
		paramErr += concat("socBIT",  socBIT_errors) + "\n";
		paramErr += concat("bSTE", bSTE_errors);
		paramErr += concat("STE",  ste_errors) + "\n" ;		
		return paramErr;
	}
}
