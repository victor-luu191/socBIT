package core;

import helpers.Updater;
import helpers.UtilFuncs;

import java.io.IOException;

import myUtil.Savers;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;
import defs.InvalidModelException;
import defs.ParamModelMismatchException;

public class Trainer {
	
	private static final double EPSILON = 1;
//	private static final double INVERSE_STEP = 0.5;
	private static final double GAMMA = Math.pow(10, -4);
	private static final double EPSILON_STEP = Math.pow(2, -20);
	
	Dataset ds;
	
	// settings of this trainer
	String model;
	int numTopic;
	Hypers hypers;
	private int maxIter;
	private double stepSize;
	private RecSysCal calculator;
	
	public Trainer(String model, Dataset ds, int numTopic, Hypers hypers, int maxIter) throws InvalidModelException {
		this.model = model;
		this.ds = ds;
		this.numTopic = numTopic;
		this.hypers = hypers;
		this.maxIter = maxIter;
		stepSize = 1;	// initial stepsize
		calculator = buildCalculator(model);
	}
	
	/**
	 * @param initParams
	 * @param resDir
	 * @return local optimal parameters which give a local minimum of the objective function (minimum errors + regularizers)
	 * @throws IOException, InvalidModelException and  ParamModelMismatchException
	 */
	Params gradDescent(Params initParams, String resDir) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		printStartMsg();
		int numIter = 0;
//		StringBuilder sbParams = new StringBuilder("iter, ...");
		StringBuilder sbObjValue = new StringBuilder("iter, obj_value \n");
		
		Params cParams = buildParams(initParams, model);
		double cValue = calculator.objValue(initParams);
		sbObjValue = sbObjValue.append(numIter + "," + cValue + "\n");
		System.out.println(numIter + ", " + cValue);
		double difference = Double.POSITIVE_INFINITY;
		
		GradCal gradCal = new GradCal(this);
		// while not convergence and still can try more
		while ( isLarge(difference) && (numIter < maxIter) ) {
			numIter ++;
			Params cGrad = gradCal.calculate(cParams, this.model);
			Params nParams = lineSearch(cParams, cGrad, cValue);
			double nValue = calculator.objValue(nParams);
			sbObjValue = sbObjValue.append(numIter + "," + nValue + "\n");
			difference = nValue - cValue;
			
			// prep for next iter
			cParams = buildParams(nParams, model);						
			cValue = nValue;
			System.out.println(numIter + "," + cValue);
		}
		
		if (!isLarge(difference)) {
			printConvergeMsg();
			String fout = resDir + "obj_values.csv";
			Savers.save(sbObjValue.toString(), fout);
		} else {
			System.out.println("Not converged yet but already exceeded the maximum number of iterations. Training stopped!!!");
		}
		
		return cParams;
	}

	private RecSysCal buildCalculator(String model) throws InvalidModelException {
		
		if (model.equalsIgnoreCase("socBIT")) {
			return new SocBIT_Cal(ds, hypers);
		} else {
			if (model.equalsIgnoreCase("STE")) {
				return new STE_Cal(ds, hypers);
			} else {
				throw new InvalidModelException();
			}
		}
	}

	private void printStartMsg() {
		System.out.println("Start training...");
		System.out.println("Iter, Objective value");
	}

	private void printConvergeMsg() {
		System.out.println("Converged to a local minimum :)");
		System.out.println("Training done.");
		System.out.println();
	}

	private boolean isLarge(double difference) {
		return Math.abs(difference) > EPSILON;
	}
	
	private Params lineSearch(Params cParams, Params cGrad, double cValue) throws ParamModelMismatchException, InvalidModelException {
		
		Params nParams = buildParams(cParams, this.model);
		boolean sufficentReduction = false;
		
		System.out.println("function diff, squared params diff, necessary reduction amount" );
		while (!sufficentReduction && (stepSize > EPSILON_STEP)) {
			stepSize = stepSize/2 ;
			nParams = Updater.update(cParams, stepSize, cGrad, this.model);
			// todo: may need some projection here to guarantee some constraints
			double nValue = calculator.objValue(nParams);
			double funcDiff = nValue - cValue;
			double sqParamDiff = sqDiff(nParams, cParams);
			double reduction = - GAMMA/stepSize * sqParamDiff;
			
			System.out.println(funcDiff + ", " + sqParamDiff + "," + reduction);
			
			sufficentReduction = (funcDiff < reduction);
			
			if (funcDiff == 0) {
				System.out.println("Meet a local minimum !!!");
				return nParams;
			}
		}
		
		if (sufficentReduction) {
//			System.out.println("Found new params with sufficient reduction");
			return nParams;
		} else {
			System.out.println("Cannot find better new params  (i.e. with sufficient reduction). "
								+ "Line search stopped due to step size too small");
			
			return cParams;
		}
	}

	private Params buildParams(Params cParams, String model) throws ParamModelMismatchException, InvalidModelException {
		
		if (model.equalsIgnoreCase("STE")) {
			return cParams;
		} 
		else {
			if (model.equalsIgnoreCase("socBIT")) {
				if (cParams instanceof SocBIT_Params) {
					return (SocBIT_Params) cParams;
				} 
				else {
					String msg = "Input params type and model mismatch!!!";
					throw new ParamModelMismatchException(msg);
				}
			} 
			else {
				throw new InvalidModelException();
			}
		}
	}

	// wrapper for computing squared difference bw two parameters where the computation depends on specific model
	private double sqDiff(Params p1, Params p2) throws InvalidModelException {

		if (model.equalsIgnoreCase("socBIT")) {
			SocBIT_Params cast_p1 = (SocBIT_Params) p1;
			SocBIT_Params cast_p2 = (SocBIT_Params) p2;
			return cast_p1.sqDiff(cast_p2);
		} else {
			if (model.equalsIgnoreCase("STE")) {
				return p1.topicDiff(p2);
			} else {
				throw new InvalidModelException();
			}
		}
		
	}
}
