package ml.loliloli;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SVDPP {
    private final int itemCount;
    private final int userCount;
    private final int f;
    private final INDArray UIM;
    private INDArray action;
    private INDArray bi, bu, user, item, y;
    private final List<History> historyList;
    private final double mu;


    private class Y{
        public INDArray sumY;
        public INDArray Ni;
    }

    private class Loss{
        public INDArray error;
        public double loss;
    }

    private static class History{
        public double loss;
        public int epoch;

        public History(double loss, int epoch) {
            this.loss = loss;
            this.epoch = epoch;
        }

        @Override
        public String toString() {
            return String.format("epoch: %d, rmse loss: %f", epoch, loss);
        }
    }

    public SVDPP(int[][] UIM, int f) {
        this.UIM = Nd4j.createFromArray(UIM); // 创建用户-物品矩阵
        userCount = this.UIM.rows();
        itemCount = this.UIM.columns();
        prepareMat();
        this.f = f;
        historyList = new ArrayList<>();
        mu = ((double) this.UIM.sumNumber()) / ((double) action.sumNumber());
    }

    private void prepareMat() {
        int row = UIM.rows();
        int col = UIM.columns();
        int[][] action_temp = new int[row][col];
        for (int r = 0; r < row; ++r) {
            for (int c = 0; c < col; ++c) {
                var val = UIM.getInt(r, c);
                if (val > 0) {
                    action_temp[r][c] = 1;
                }else{
                    action_temp[r][c] = 0;
                }
            }
        }
        action = Nd4j.createFromArray(action_temp);
    }

    public void fit(int epoch, double lr, double ku, double ki, boolean debug, INDArray eval){
        bi = Nd4j.rand(1, itemCount);
        bu = Nd4j.rand(userCount, 1);

        user = randomInit(-0.01, 0.01, userCount, f);
        item = randomInit(-0.01, 0.01, itemCount, f);

        y = Nd4j.rand(itemCount, f);

        for(var ep = 0; ep < epoch; ep++){
            Y sumY = getY();
            INDArray predict = ((user.add(sumY.sumY)).mmul(item.transpose())).add(bu).add(bi).add(mu);
            computeLoss(predict, UIM, action);
            Loss loss = computeLoss(predict, UIM, action);
            updateParam(loss, lr, ku, ki, sumY);
            History history = new History(loss.loss, ep);
            historyList.add(history);
            if (debug){
                System.out.print(history);
                if ((ep + 1) % 10 == 0){
                    var evalLoss = eval(eval);
                    System.out.printf(" eval rmse loss: %f%n", evalLoss);
                }else{
                    System.out.print("\n");
                }
            }
        }
    }

    private Loss computeLoss(INDArray predict,INDArray UIM, INDArray action) {
        double[][] pred_temp = predict.castTo(DataType.DOUBLE).toDoubleMatrix();
        for (var i = 0; i < pred_temp.length; ++i){
            for (var j = 0; j < pred_temp[i].length; ++j){
                if (pred_temp[i][j] > 5){
                    pred_temp[i][j] = 5;
                }else if (pred_temp[i][j] < 1){
                    pred_temp[i][j] = 1;
                }
            }
        }
        predict = Nd4j.createFromArray(pred_temp);
        return rmse(predict, UIM, action);
    }

    public double eval(INDArray groundTruth){
        Y sumY = getY();
        double[][] evalAction = new double[groundTruth.rows()][groundTruth.columns()];
        for (var i = 0; i < evalAction.length; ++i){
            for (var j = 0; j < evalAction[i].length; ++j){
                if (groundTruth.getDouble(i, j) > 0){
                    evalAction[i][j] = 1;
                }else{
                    evalAction[i][j] = 0;
                }
            }
        }
        INDArray evalAct = Nd4j.createFromArray(evalAction);
        var predict = ((user.add(sumY.sumY)).mmul(item.transpose())).add(bu).add(bi).add(mu);
        //predict.muli(evalAct);
        Loss loss = computeLoss(predict, groundTruth, evalAct);
        return loss.loss;
    }

    private INDArray randomInit(double min, double max, int ...shape){
        double[][] temp = new double[shape[0]][shape[1]];
        var rand = new Random();
        for (var i = 0; i < shape[0]; ++i){
            for (var j = 0; j < shape[1]; ++j){
                var val = rand.nextDouble();
                while (val > max || val < min){
                    val = rand.nextDouble();
                }
                temp[i][j] = val;
            }
        }
        return Nd4j.createFromArray(temp);
    }

    private Y getY(){
        Y y = new Y();
        y.Ni = action.sum(1);
        double[][] temp_y = new double[userCount][1];
        for (var uid = 0; uid < action.rows(); ++uid){
            double userY = 0;
            double count = 0;
            INDArray line = action.getRow(uid);
            int c = 0;
            for (var i : line.toDoubleVector()){
                int a = 0;
                if (i != 0){
                    for (var j : this.y.getRow(c).toDoubleVector()){
                        userY += j;
                        count += 1;
                    }
                }
                c += 1;
            }
            double sunUserY = userY / count;
            if (sunUserY == 0){
                temp_y[uid][0] = 0.1;
            }else{
                temp_y[uid][0] = sunUserY / y.Ni.getDouble(uid);
            }
        }
        y.sumY = Nd4j.createFromArray(temp_y);
        return y;
    }

    private Loss rmse(INDArray predict, INDArray UIM, INDArray action){
        Loss e = new Loss();
        e.error = UIM.sub(predict);
        e.loss = (double) (e.error.mul(e.error).mul(action)).sumNumber() / (double) action.sumNumber();
        e.error = e.error.mul(action);
        return e;
    }

    private void updateParam(Loss loss, double lr, double ku, double ki, Y sumY){

        double[] sumI_tmp = action.sum(0).toDoubleVector();
        double[] sumU_tmp = action.sum(1).toDoubleVector();
        for (var i = 0; i < sumI_tmp.length; ++i){
            if (sumI_tmp[i] == 0){
                sumI_tmp[i] += 1;
            }
        }
        for (var i = 0; i < sumU_tmp.length; ++i){
            sumU_tmp[i] = sumU_tmp[i] == 0 ? 1 : sumU_tmp[i];
        }
        INDArray sumI = Nd4j.createFromArray(sumI_tmp);
        INDArray sumU = Nd4j.createFromArray(sumU_tmp);
        INDArray errorI = loss.error.castTo(DataType.DOUBLE).sum(0).div(sumI);
        INDArray errorU = loss.error.castTo(DataType.DOUBLE).sum(1).div(sumU);
        errorU = errorU.reshape(userCount, 1);

        bi = bi.add(errorI.sub(bi.mul(ki)).mul(lr));
        bu = bu.add(errorU.sub(bu.mul(ku)).mul(lr));

        INDArray ugrad = loss.error.castTo(DataType.DOUBLE).mmul(item.castTo(DataType.DOUBLE)).sub(user.mul(ku)).mul(lr);
        INDArray igrad = loss.error.castTo(DataType.DOUBLE).transpose().mmul(user.castTo(DataType.DOUBLE).add(sumY.sumY)).sub(item.mul(ki)).mul(lr);

        user.addi(ugrad);
        item.addi(igrad);

        for (var i = 0; i < action.rows(); ++i){
            INDArray rating_list = action.getRow(i);
            y.get(rating_list)
                    .add(
                            Nd4j.tile(loss.error.getRow(i).get(rating_list).reshape(itemCount, 1), 1, 30)
                                    .mul(
                                            item.get(rating_list).div(sumY.Ni.getDouble(i)
                                            ).sub(
                                                    y.get(rating_list).mul(ku)
                                            )
                                    ).mul(lr)
                    );
        }
    }

    public List<History> getHistoryList(){
        return historyList;
    }
}
