import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.clusterers.SimpleKMeans;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Properties;

import weka.clusterers.ClusterEvaluation;

/**
 * 
 * @author jason
 *
 */
public class KMeans {

	public static void main(String[] args) {
        // TODO code application logic here
		try{
			InputStream input = new FileInputStream("infos.properties");
			Properties prop = new Properties();
			// load a properties file
			prop.load(input);
			
            DataSource src = new DataSource(prop.getProperty("trainingFile"));
            Instances dataset = src.getDataSet();
            SimpleKMeans model = new SimpleKMeans();
            
            
            model.setPreserveInstancesOrder(true);
            model.setNumClusters(Integer.valueOf(prop.getProperty("clusters")));
            model.buildClusterer(dataset);
            System.out.println("DataPoints in each cluster:");
    		for (Instance instance : dataset) {
                System.out.println( 
                        instance.toString()+": "+
                        model.clusterInstance(instance));
            }
 
            ClusterEvaluation eval = new ClusterEvaluation();
            DataSource src1 = new DataSource(prop.getProperty("evaluationFile"));
            Instances tdt = src1.getDataSet();
            eval.setClusterer(model);
            eval.evaluateClusterer(tdt);
            System.out.println(eval.clusterResultsToString());
		}
        catch(Exception e)
        {
            System.out.println(e.getMessage());
        }
    }
}
