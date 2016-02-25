package core;

import helpers.DataLoader;
import helpers.ParamLoader;
import helpers.ParamSaver;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

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
		String errStr = "numTopic, topicUserErr, topicItemErr, brandUserErr, brandItemErr, decisionPrefErr \n";
		Params socBIT_params = null;
		
		int minK = 5; int maxK = 15;
		for (int numTopic = minK; numTopic <=  maxK; numTopic++) {
			
			socBIT_params = trainBySocBIT(ds, numTopic);
			if (numTopic == 10) {// 10 is currently the ground-truth number of topics
				Errors errors = compDiff( (SocBIT_Params) socBIT_params, (SocBIT_Params) gt_params);
				errStr += concat(numTopic, errors) + "\n";
			}
			
//			Params ste_params = trainBySTE(ds, numTopic);
		}	
		
		String fErrors = "result/syn/N" + ds.numUser + "/unif/" + "/" ;
		Savers.save(errStr, fErrors);
		String resDir = "result/syn/N" + numUser + "/unif/";
		ParamSaver.save(socBIT_params, resDir);
		
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
	
	private static String concat(int numTopic, Errors errors) {
		return numTopic + ","  + 	errors.topicUser + "," + errors.topicItem + "," + 
									errors.brandUser + "," + errors.brandItem + "," + 
									errors.decisionPrefs ;
	}

	private static Errors compDiff(SocBIT_Params params1, SocBIT_Params params2) {
		
		double topicUserErr = params1.topicUser.subtract(params2.topicUser).getFrobeniusNorm();
		double topicItemErr = params1.topicItem.subtract(params2.topicItem).getFrobeniusNorm();
		double brandUserErr = params1.brandUser.subtract(params2.brandUser).getFrobeniusNorm();
		double brandItemErr = params1.brandItem.subtract(params2.brandItem).getFrobeniusNorm();
		double decisionPrefErr = toVector(params1.userDecisionPrefs).subtract(toVector(params2.userDecisionPrefs)).getNorm();
		
		return new Errors(topicUserErr, topicItemErr, brandUserErr, brandItemErr, decisionPrefErr);
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
