import ml.loliloli.SVDPP;
import org.bytedeco.opencv.presets.opencv_core;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Recommend {
    public static void main(String[] args) throws Exception{

        File dataInfo = new File("dataset/u.info");
        InputStreamReader dataInfoStream = new InputStreamReader(
                new FileInputStream(dataInfo)
        );
        BufferedReader dataInfoReader = new BufferedReader(
                dataInfoStream
        );
        String line = dataInfoReader.readLine();
        List<Integer> data = new ArrayList<>();
        while (line != null){
            String[] records = line.split(" ");
            data.add(Integer.parseInt(records[0]));
            line = dataInfoReader.readLine();
        }
        int userCount = data.get(0);
        int itemCount = data.get(1);
        System.out.printf("user: %d, item: %d%n", userCount, itemCount);
        int[][] UIM_temp = new int[userCount][itemCount];
        int[][] UIM_eval = new int[userCount][itemCount];
        int train = 0;
        int eval = 0;
        File dataSource = new File("dataset/u.data");
        InputStreamReader dataSourceStream = new InputStreamReader(
                new FileInputStream(dataSource)
        );
        BufferedReader dataSourceReader = new BufferedReader(dataSourceStream);
        line = dataSourceReader.readLine();
        Random rand = new Random();
        while (line != null){
            String[] records = line.split("\t");
            int user = Integer.parseInt(records[0]) - 1;
            int item = Integer.parseInt(records[1]) - 1;
            int rate = Integer.parseInt(records[2]);
            if (rand.nextDouble() > 0.8){
                UIM_eval[user][item] = rate;
                eval += 1;
            }else{
                UIM_temp[user][item] = rate;
                train += 1;
            }
            line = dataSourceReader.readLine();
        }
        System.out.printf("Total %d samples, %d train, %d eval%n", train+eval, train, eval);
        System.out.println("data loading finished, start training");


        INDArray UIM = Nd4j.createFromArray(UIM_temp);
        INDArray UIMEval = Nd4j.createFromArray(UIM_eval);
        SVDPP svdpp = new SVDPP(UIM.toIntMatrix(), 30);

        svdpp.fit(200, 0.001, 0.1, 0.1, true, UIMEval);
    }
}
