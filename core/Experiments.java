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
import defs.ParamModelMismatchException;

public class Experiments {
	
	static Dataset train_ds;
	static Dataset test_ds;
	
	public static void main(String[] args) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		// temporary value passing, later will read from file data_stats or itemInfo
		int numUser = 1000;
		int numBrand = 20;	

		// TODO: switch to the pkg org.kohsuke.args4j to enable named args
		String dataDir = "data/syn/N" + numUser + "/unif/";	// or args[0]
		Dataset ds = DataLoader.load(dataDir, numBrand);
		double train_ratio = 0.8;
		split(ds, train_ratio);
		
		
		String gtParamDir = dataDir + "true_params/";
		Params gt_params = ParamLoader.load(gtParamDir);
		String errStr = "model, numTopic, topicUserErr, topicItemErr, brandUserErr, brandItemErr, decisionPrefErr \n";
		Params socBIT_params = null;
		
		int minK = 5; int maxK = 15;
		for (int numTopic = minK; numTopic <=  maxK; numTopic++) {
			
			socBIT_params = trainBySocBIT(ds, numTopic);
			Params ste_params = trainBySTE(ds, numTopic);
			
			if (numTopic == 10) {// 10 is currently the ground-truth number of topics
				Errors socBIT_errors = compDiff( (SocBIT_Params) socBIT_params, (SocBIT_Params) gt_params);
				Errors ste_errors = compDiff(ste_params, gt_params);
				errStr += concat("socBIT", numTopic, socBIT_errors) + "\n";
				errStr += concat("STE", numTopic, ste_errors) + "\n" ;
			}
		}	
		
		String resDir = "result/syn/N" + numUser + "/unif/";
		ParamSaver.save(socBIT_params, resDir);
		
		String fErrors = resDir + "errors.csv" ;
		Savers.save(errStr, fErrors);
		
		predict(socBIT_params, test_ds);
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
	
	private static Params trainBySocBIT(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		System.out.println("Training by socBIT model");
		
		String resDir = "result/syn/N" + ds.numUser + "/socBIT/numTopic" + numTopic + "/" ; // "/unif/numTopic" + numTopic + "/"
		if (!Files.exists(Paths.get(resDir))) {
			Files.createDirectories(Paths.get(resDir));
		} 
		
		Trainer trainer = initTrainer("socBIT", ds, numTopic);	// currently training on whole data set, switch to training set later	
		SocBIT_Params initParams = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, trainer.numTopic);
		Params learned_params = trainer.gradDescent(initParams, resDir);
//		save(learned_params, resDir);
		return learned_params;
	}
	
	@SuppressWarnings("unused")
	private static Params trainBySTE(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		System.out.println("Training by STE model...");
		
		String resDir = "result/syn/N" + ds.numUser + "/STE/numTopic" + numTopic + "/" ; // "/unif/numTopic" + numTopic + "/"
		if (!Files.exists(Paths.get(resDir))) {
			Files.createDirectories(Paths.get(resDir));
		} 
		
		Trainer trainer = initTrainer("STE", ds, numTopic);	// currently training on whole data set, switch to training set later	
		Params initParams = new Params(ds.numUser, ds.numItem, trainer.numTopic);
		initParams.createFeatsUniformly();
		Params learned_params = trainer.gradDescent(initParams, resDir);
//		save(learned_params, resDir);
		return learned_params;
	}
	
	/**
	 * Make predictions using specified {@link params} and check with ground truth {@link test_ds} to obtain ... 
	 * evaluation metrics
	 * @param params
	 * @param test_ds
	 */
	private static void predict(Params params, Dataset test_ds) {
		// TODO Auto-generated method stub
	}
	
	private static String concat(String model, int numTopic, Errors errors) {
		return model + "," + numTopic + ","  + 	errors.toString();
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
			return new Errors(topicUserErr, topicItemErr, Optional.empty(), Optional.empty(), Optional.empty());
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

}
