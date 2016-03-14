package core;

import helpers.DataLoader;
import helpers.ParamLoader;
import helpers.ParamSaver;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Optional;

import myUtil.Savers;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Errors;
import defs.Hypers;
import defs.InvalidModelException;
import defs.NonConvergeException;
import defs.ParamModelMismatchException;
import defs.Params;
import defs.Result;
import defs.SocBIT_Params;

public class Experiment {
	
	static Dataset train_ds;
	static Dataset test_ds;
	
	public static void main(String[] args) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		// temporary value passing, later will read from file data_stats or itemInfo
		int numUser = 1000;
		int gt_numTopic = 5;
		int numBrand = 9*gt_numTopic + 1;	

		// TODO: switch to the pkg org.kohsuke.args4j to enable named args
		String dataDir = "data/syn/N" + numUser + "/unif/";	// or args[0]
		Dataset ds = DataLoader.load(dataDir, numBrand);
		double train_ratio = 0.8;
		split(ds, train_ratio);
		
		
		String gtParamDir = dataDir + "true_params/";
		Params gt_params = ParamLoader.load(gtParamDir);
		
		String resDir = "result/syn/N" + ds.numUser + "/";
		String errDir = resDir + "errors/"; 		mkDir(errDir);
		
		
		String allErrStr = "model, numTopic, ratingErr, edgeWeightErr, obj_value" + "\n";
		
		int minK = 1; int maxK = 10;
//		int minK = gt_numTopic; int maxK = gt_numTopic;	// for fast testing
		for (int numTopic = minK; numTopic <=  maxK; numTopic++) {
			Result socBIT_result = trainBySocBIT(ds, numTopic);
			Result ste_result = trainBySTE(ds, numTopic);
			allErrStr += "socBIT, " + numTopic + "," + socBIT_result.toErrString() + "\n";
			allErrStr += "STE, " + numTopic + "," + ste_result.toErrString() + "\n";
			
			SocBIT_Params socBIT_params = (SocBIT_Params) socBIT_result.learnedParams;
			Params ste_params = ste_result.learnedParams;
			String model = "socBIT";
			save(socBIT_params, model, numTopic, resDir);
			model = "STE";
			save(ste_params, model, numTopic, resDir);
			
			if (numTopic == gt_numTopic) {
				String paramErr = getParamErr(socBIT_params, ste_params, gt_params);
				String fParamErr = errDir + "param_recover.csv" ;
				Savers.save(paramErr, fParamErr);
			}
		}	
		
		String fErrors = errDir + "all_errors.csv";
		Savers.save(allErrStr, fErrors);
		
		
//		predict(socBIT_params, test_ds);
	}
	
	/**
	 * split full ds into {@link training_ds} and {@link test_set}, with {@link training_ds} occupy a ratio {@link train_ratio} 
	 * @param ds
	 * @param train_ratio 
	 */
	private static void split(Dataset ds, double train_ratio) {
		// TODO Auto-generated method stub
		
	}
	
	private static Trainer initTrainer(String model, Dataset ds, int numTopic) throws InvalidModelException {
		
		int maxIter = 100;
		
		Hypers hypers = null;
		if (model.equalsIgnoreCase("socBIT")) {
			hypers = Hypers.assignBySocBIT();
			System.out.println("Try " + numTopic + " topics. Start training...");
//			printRegConst(hypers);
		} 
		else {
			if (model.equalsIgnoreCase("STE")) {
				hypers = Hypers.assignBySTE();
				System.out.println("Try " + numTopic + " topics.");
			} else {
				throw new InvalidModelException();
			}
		}
		
		Trainer trainer = new Trainer(model, ds, numTopic, hypers, maxIter);
		return trainer;
	}
	
	private static Result trainBySocBIT(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		System.out.println("Training by socBIT model");
		
		Trainer trainer = initTrainer("socBIT", ds, numTopic);	// currently training on whole data set, switch to training set later	
		SocBIT_Params initParams = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, trainer.numTopic);
		Result result = trainer.gradDescent(initParams);
		return result;
	}
	
	private static Result trainBySTE(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		System.out.println("Training by STE model...");
		
		Trainer trainer = initTrainer("STE", ds, numTopic);	// currently training on whole data set, switch to training set later	
		Params initParams = new Params(ds.numUser, ds.numItem, trainer.numTopic);
		initParams.createFeatsUniformly();
		Result result = trainer.gradDescent(initParams);
		
		return result;
	}
	
	/**
	 * Make predictions using specified {@link params} and check with ground truth {@link test_ds} to obtain ... 
	 * evaluation metrics
	 * @param params
	 * @param test_ds
	 */
	@SuppressWarnings("unused")
	private static void predict(Params params, Dataset test_ds) {
		// TODO Auto-generated method stub
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
	
	private static void save(Params params, String model, int numTopic, String resDir) throws IOException {
		
		String name = resDir + model + "/" + "numTopic" + numTopic + "/";
		mkDir(name);
		ParamSaver.save(params, name);
	}

	private static void mkDir(String name) throws IOException {
		if (!Files.exists(Paths.get(name))) {
			Files.createDirectories(Paths.get(name));
		}
	}

	private static String getParamErr(SocBIT_Params socBIT_params, Params ste_params, Params gt_params) {
		
		String paramErr = "model, topicUserErr, topicItemErr, brandUserErr, brandItemErr, decisionPrefErr \n";
		Errors socBIT_errors = compDiff( socBIT_params, (SocBIT_Params) gt_params);
		Errors ste_errors = compDiff(ste_params, gt_params);
		paramErr += concat("socBIT",  socBIT_errors) + "\n";	
		paramErr += concat("STE",  ste_errors) + "\n" ;		
		return paramErr;
	}
}
