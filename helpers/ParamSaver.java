package helpers;

import java.io.IOException;
import java.util.Arrays;

import myUtil.Savers;
import core.Params;
import core.SocBIT_Params;

public class ParamSaver {
	
	public static void save(Params params, String resDir) throws IOException {
		
		if (params instanceof SocBIT_Params) {
			saveTopicComponents(params, resDir);
			saveBrandComponents(params, resDir);
		}
		else {
			saveTopicComponents(params, resDir);
		}
	}

	private static void saveBrandComponents(Params params, String resDir)
			throws IOException {
		SocBIT_Params castParams = (SocBIT_Params) params;
		String userBrandFeat_file = resDir + "user_brand_feats.csv";
		Savers.save(castParams.brandUser.toString(), userBrandFeat_file);
		
		String itemBrandFeat_file = resDir + "item_brand_feats.csv";
		Savers.save(castParams.brandItem.toString(), itemBrandFeat_file);
		
		String decisionPref_file = resDir + "decision_prefs.csv";
		Savers.save(Arrays.toString(castParams.userDecisionPrefs), decisionPref_file);
	}

	private static void saveTopicComponents(Params params, String resDir)
			throws IOException {
		String userTopicFeat_file = resDir + "user_topic_feats.csv";
		Savers.save(params.topicUser.toString(), userTopicFeat_file);
		
		String itemTopicFeat_file = resDir + "item_topic_feats.csv";
		Savers.save(params.topicItem.toString(), itemTopicFeat_file);
	}
}
