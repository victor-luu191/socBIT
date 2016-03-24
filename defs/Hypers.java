package defs;

public class Hypers {
	
	public double topicLambda; 
	public double brandLambda;
	public double weightLambda;
	public double decisionLambda;
	
	public double alpha;	// tuning param only for STE and brandSTE
							// alpha controls how much a user believe in his own ratings vs. friend's ratings
	
	public Hypers(double topicLambda, double weightLambda) {
		this.topicLambda = topicLambda;
		this.weightLambda = weightLambda;
	}
	
	public Hypers(double topicLambda, double brandLambda, double weightLambda, double decisionLambda) {
		
		this.topicLambda = topicLambda;
		this.brandLambda = brandLambda;
		this.weightLambda = weightLambda;
		this.decisionLambda = decisionLambda;
	}

	public static Hypers setBySoRec(double topicLambda, double weightLambda) {
		return new Hypers(topicLambda,  weightLambda);
	}
	
	public static Hypers setBySocBIT(double topicLambda, double brandLambda, double weightLambda, double decisionLambda) {
		return new Hypers(topicLambda, brandLambda, weightLambda, decisionLambda);
	}

	
//	public Hypers(double topicLambda, double alpha) {
//	this.topicLambda = topicLambda;
//	this.alpha = alpha;
//}

//public Hypers(double topicLambda, double brandLambda, double weightLambda, double decisionLambda, double alpha) {
//
//this.topicLambda = topicLambda;
//this.brandLambda = brandLambda;
//this.weightLambda = weightLambda;
//this.decisionLambda = decisionLambda;
//this.alpha = alpha;
//}
	
//	private static Hypers setBySTE(double topicLambda, double alpha) {
//	return new Hypers(topicLambda, alpha);
//}
//
//private static Hypers setByBSTE(double topicLambda, double brandLambda, double weightLambda, double decisionLambda, double alpha) {
//	
//	return new Hypers(topicLambda, brandLambda, weightLambda, decisionLambda, alpha);
//}
	
//	public static Hypers assignBySTE() {
//		double topicLambda = 10;
//		double alpha = 0.5;
//		return new Hypers(topicLambda, alpha);
//	}
//	
//	public static Hypers assignByBSTE() {
//		double topicLambda = 1;
//		double brandLambda = 0.5;
//		double weightLambda = 0.001;
//		double decisionLambda = 0.1;
//		double alpha = 0.5;
//		Hypers hypers = setByBSTE(topicLambda, brandLambda, weightLambda, decisionLambda, alpha);
//		return hypers;
//	}

	

	
}
