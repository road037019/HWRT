#include "predict.h"
#include "allocate.h"

struct PhyInfo{
    int phycpu;
    int phymem;
    int phyhard;
} phyinfo;
struct FlavorsInfo{
    vector<string> vflavors;
    vector<int> vcpus, vmems;
} flavorsinfo;

int predict_daySpan, history_daySpan;
vector<int> vflavors_pridict_nums;
string dim, predict_begin_time, predict_end_time, history_begin_time, history_end_time;
vector< vector<int> > sequences;
time_t predict_begin_time_t, predict_end_time_t, history_begin_time_t, history_end_time_t;
int oneDayLong = 24 * 3600; //s

void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{
    get_info(info);

    get_data(data, data_num);
    my_predict();

    allocate_simple(flavorsinfo.vflavors, vflavors_pridict_nums, dim, filename);

//	char * result_file = (char *)"17\n\n0 8 0 20";
//	write_result(result_file, filename);
}
void my_predict()
{
    // init sequences
    vector< vector<int> > daySpan_sequences;
    for(int i=0; i<sequences.size(); i++)
    {
        vector<int> daySpan_sequence;
        //累加predict_daySpan内的记录数
        for(int j=0; j<sequences[i].size()-predict_daySpan+1; j++)
        {
            daySpan_sequence.push_back(accumulate(sequences[i].begin()+j, sequences[i].begin()+j+predict_daySpan, 0));
        }
        daySpan_sequences.push_back(daySpan_sequence);
    }
    cout << "daySpan_sequences: " << endl;
    for(int i=0; i<daySpan_sequences.size(); i++)
    {
        cout << flavorsinfo.vflavors[i] << endl;
        for(int j=0; j<daySpan_sequences[i].size(); j++)
        {
            cout << daySpan_sequences[i][j] << ", ";
        }
        cout << endl;
    }

    for(int i=0; i<daySpan_sequences.size(); i++)
    {
        vector<int> temp = daySpan_sequences[i];  //temp for daySpan_sequences
        for(int j=0; j<predict_daySpan; j++)
        {
            int sum = accumulate(temp.begin(), temp.end(), 0);
            float average = (float)sum/temp.size();
            float last_daySpan = temp.back();


            float w_last_daySpan = 1;
            float w_average = 1- w_last_daySpan;
            int predict_result = round(w_average * average + w_last_daySpan * last_daySpan);
//            cout << predict_result;
            temp.push_back(predict_result);
        }
        vflavors_pridict_nums.push_back(temp.back());
    }

    for(int i=0; i<flavorsinfo.vflavors.size(); i++)
    {
        cout << flavorsinfo.vflavors[i] << ":\t" << vflavors_pridict_nums[i] << endl;
    }
}
void my_predict_w_trained()
{
    // init sequences
    vector< vector<int> > daySpan_sequences;
    for(int i=0; i<sequences.size(); i++)
    {
        vector<int> daySpan_sequence;
        //累加predict_daySpan内的记录数
        for(int j=0; j<sequences[i].size()-predict_daySpan+1; j++)
        {
            daySpan_sequence.push_back(accumulate(sequences[i].begin()+j, sequences[i].begin()+j+predict_daySpan, 0));
        }
        daySpan_sequences.push_back(daySpan_sequence);

        // train weights
        int steps = 100;
        for(int step=0; step<steps; step++)
        {
            float w_last_daySpan = 1.0 * step/steps;
            float predict_error = train_w(daySpan_sequence, w_last_daySpan);
            cout << "w_last_daySpan: " << w_last_daySpan << "\terror: " << predict_error << endl;
        }

    }
}
float train_w(vector<int> daySpan_sequence, float w_last_daySpan)
{
    vector<float> daySpan_predict_error_sequence;
    //history data: 0~i
    for(int i=daySpan_sequence.size()/2; i<daySpan_sequence.size()-1; i++)
    {
        //last_daySpan
        float last_daySpan = daySpan_sequence[i];
        //average
        int sum = 0;
        for(int j=0; j<=i; j++)
        {
            sum += daySpan_sequence[j];
        }
        float average = (float)sum / (i+1);
        float predict_result = w_last_daySpan * last_daySpan + (1-w_last_daySpan) * average;

        //raw data:      daySpan_sequence[i+1]
        //predict data:  predict_result
        //predict error: |raw data - predict data|
        daySpan_predict_error_sequence.push_back(fabs((float)daySpan_sequence[i+1] - predict_result));
    }

    float sum_error = accumulate(daySpan_predict_error_sequence.begin(), daySpan_predict_error_sequence.end(), 0.0);
    return sum_error/daySpan_predict_error_sequence.size();
}
void gray_predict()
{
    // init sequences A
    vector< vector<int> > gray_sequences;
    for(int i=0; i<sequences.size(); i++)
    {
        vector<int> gray_sequence;
        //累加predict_daySpan内的记录数
        for(int j=0; j<sequences[i].size()-predict_daySpan+1; j++)
        {
            gray_sequence.push_back(accumulate(sequences[i].begin()+j, sequences[i].begin()+j+predict_daySpan, 0));
        }
        gray_sequences.push_back(gray_sequence);
    }
    cout << "gray_sequences: " << endl;
    for(int i=0; i<gray_sequences.size(); i++)
    {
        cout << flavorsinfo.vflavors[i] << endl;
        for(int j=0; j<gray_sequences[i].size(); j++)
        {
            cout << gray_sequences[i][j] << ", ";
        }
        cout << endl;
    }

    // cumsum
    vector< vector<int> > B_sequences;
    for(int i=0; i<gray_sequences.size(); i++)
    {
        vector<int> B_sequence;

        for(int j=0; j<gray_sequences[i].size(); j++)
        {
            B_sequence.push_back(accumulate(gray_sequences[i].begin(), gray_sequences[i].begin()+j+1, 0));
        }
        B_sequences.push_back(B_sequence);
    }
    cout << "B_sequences: " << endl;
    for(int i=0; i<B_sequences.size(); i++)
    {
        cout << flavorsinfo.vflavors[i] << endl;
        for(int j=0; j<B_sequences[i].size(); j++)
        {
            cout << B_sequences[i][j] << ", ";
        }
        cout << endl;
    }

    // generate
    vector< vector<int> > C_sequences;
    for(int i=0; i<B_sequences.size(); i++)
    {
        vector<int> C_sequence;

        for(int j=0; j<B_sequences[i].size()-1; j++)
        {
            C_sequence.push_back((B_sequences[i][j] + B_sequences[i][j+1])/2);
        }
        C_sequences.push_back(C_sequence);
    }
    cout << "C_sequences: " << endl;
    for(int i=0; i<C_sequences.size(); i++)
    {
        cout << flavorsinfo.vflavors[i] << endl;
        for(int j=0; j<C_sequences[i].size(); j++)
        {
            cout << C_sequences[i][j] << ", ";
        }
        cout << endl;
    }

    //计算待定参数的值
    _Matrix_Calc m_c;
    for(int i=0; i<gray_sequences.size(); i++)
    {
        _Matrix D(gray_sequences[i].size()-1,1);
        //初始化内存
        D.init_matrix();
        //初始化数据
        for(int mm=0; mm<D.m; mm++)
        {
          D.write(mm, 0, gray_sequences[i][mm+1]);
        }
//        m_c.printff_matrix(&D);
        _Matrix E(2,C_sequences[i].size());
        E.init_matrix();
        for(int nn=0; nn<E.n; nn++)
        {
            E.write(0, nn, -C_sequences[i][nn]);
            E.write(1, nn, 1);
        }

        _Matrix E_trans(C_sequences[i].size(),2);
        E_trans.init_matrix();
        if(1!=m_c.transpos(&E, &E_trans)) cout<<"transpos error 1......";

        _Matrix E_E_trans(2,2);
        E_E_trans.init_matrix();
        if(1!=m_c.multiply(&E, &E_trans, &E_E_trans)) cout<<"multiply error 1......";

        _Matrix E_E_trans_inv(2,2);
        E_E_trans_inv.init_matrix();
        if(1!=m_c.inverse(&E_E_trans, &E_E_trans_inv)) cout<<"inverse error......";

        _Matrix E_E_trans_inv_E(2,C_sequences[i].size());
        E_E_trans_inv_E.init_matrix();
        if(1!=m_c.multiply(&E_E_trans_inv, &E, &E_E_trans_inv_E)) cout<<"multiply error 2......";

        _Matrix E_E_trans_inv_E_D(2, 1);
        E_E_trans_inv_E_D.init_matrix();
        if(1!=m_c.multiply(&E_E_trans_inv_E, &D, &E_E_trans_inv_E_D)) cout<<"multiply error 3......";

        _Matrix c(1,2);
        c.init_matrix();
        if(1!=m_c.transpos(&E_E_trans_inv_E_D, &c)) cout<<"transpos error 2......";

        float a, b;
        a = c.read(0,0);
        b = c.read(0,1);

        //预测后续数据
        int predict_times = 1;
        vector<float> F;
        F.push_back(gray_sequences[i][0]);
        for(int j=1; j<gray_sequences[i].size()+predict_times; j++)
        {
            F.push_back((F[1]-b/a)/exp(a*j) + b/a);
        }
        vector<float> G;
        G.push_back(gray_sequences[i][0]);
        for(int j=1; j<gray_sequences[i].size()+predict_times; j++)
        {
            G.push_back(F[j]-F[j-1]);
        }
    }

}
void get_info(char * info[MAX_INFO_NUM])
{
    int num_of_vm;

    string line = info[0];
    int firstSpace = line.find_first_of(' ');
    int secondSpace = line.find_last_of(' ');
    phyinfo.phycpu = atoi(line.substr(0,firstSpace).c_str());
    phyinfo.phymem = atoi(line.substr(firstSpace+1, secondSpace-firstSpace-1).c_str())*1024; //MB
    phyinfo.phyhard = atoi(line.substr(secondSpace+1).c_str());

    line = info[2];
    num_of_vm = atoi(line.c_str());

    for(int i=3; i<3+num_of_vm; i++)
    {
        line = info[i];
        int firstSpace = line.find_first_of(' ');
        int secondSpace = line.find_last_of(' ');
        flavorsinfo.vflavors.push_back(line.substr(0,firstSpace));
        flavorsinfo.vcpus.push_back(atoi(line.substr(firstSpace+1, secondSpace-firstSpace-1).c_str()));
        flavorsinfo.vmems.push_back(atoi(line.substr(secondSpace+1).c_str()));
    }   

    dim = info[3+num_of_vm+1];
    predict_begin_time = info[3+num_of_vm+3];
    predict_end_time = info[3+num_of_vm+4];
    predict_begin_time_t = str2time(predict_begin_time.c_str(), "%Y-%m-%d %H:%M:%S");
    predict_end_time_t = str2time(predict_end_time.c_str(), "%Y-%m-%d %H:%M:%S");
    predict_daySpan = ceil((float)(predict_end_time_t - predict_begin_time_t)/oneDayLong);

    //对flavors排序，flavor1~flavor15  加了这一步骤，对结果没影响
    for(int i=0; i<flavorsinfo.vflavors.size()-1; i++)
    {
        for(int j=i+1; j<flavorsinfo.vflavors.size(); j++)
        {
            if(atoi(flavorsinfo.vflavors[j].substr(sizeof("flavor")-1).c_str()) < atoi(flavorsinfo.vflavors[i].substr(sizeof("flavor")-1).c_str()))
            {
                //交换
                string vflavor_tmp;
                int vcpu_tmp, vmem_temp;

                vflavor_tmp = flavorsinfo.vflavors[i];
                vcpu_tmp    = flavorsinfo.vcpus[i];
                vmem_temp   = flavorsinfo.vmems[i];

                flavorsinfo.vflavors[i] = flavorsinfo.vflavors[j];
                flavorsinfo.vcpus[i]    = flavorsinfo.vcpus[j];
                flavorsinfo.vmems[i]    = flavorsinfo.vmems[j];

                flavorsinfo.vflavors[j] = vflavor_tmp;
                flavorsinfo.vcpus[j]    = vcpu_tmp;
                flavorsinfo.vmems[j]    = vmem_temp;
            }
        }
    }

    cout << "phycpu: " << phyinfo.phycpu << endl;
    cout << "phymem: " << phyinfo.phymem << endl;
    cout << "phyhard: " << phyinfo.phyhard << endl;
    cout << "num_of_vm: " << num_of_vm << endl;
    for(int i=0; i<flavorsinfo.vflavors.size(); i++)
    {
        cout << flavorsinfo.vflavors[i]<<" "<< flavorsinfo.vcpus[i] << " " << flavorsinfo.vmems[i]<<endl;
    }
    cout << "dim: " << dim <<endl;
    cout << "predict_begin_time: " << predict_begin_time << endl;
    cout << "predict_end_time: " << predict_end_time << endl;
    cout << "predict_daySpan: " << predict_daySpan << endl;
}
void get_data(char * data[MAX_DATA_NUM], int data_num)
{
    string line = data[0];
    int firstTab = line.find_first_of('\t');
    int secondTab = line.find_last_of('\t');
    int lastSpace = line.find_last_of(' ');
    history_begin_time = line.substr(secondTab+sizeof("\t")-1, lastSpace-secondTab-1);
    history_begin_time.append(" 00:00:00");
    cout << history_begin_time <<endl;
    line = data[data_num-1];
    firstTab = line.find_first_of('\t');
    secondTab = line.find_last_of('\t');
    lastSpace = line.find_last_of(' ');
    history_end_time = line.substr(secondTab+sizeof("\t")-1, lastSpace-secondTab-1);
    history_end_time.append(" 23:59:59");
    cout << history_end_time <<endl;

    history_begin_time_t = str2time(history_begin_time.c_str(), "%Y-%m-%d %H:%M:%S");
    cout << history_begin_time_t << endl;
    history_end_time_t = str2time(history_end_time.c_str(), "%Y-%m-%d %H:%M:%S");
    cout << history_end_time_t << endl;
    history_daySpan = ceil((float)(history_end_time_t - history_begin_time_t)/oneDayLong);
    cout << "history_daySpan: " << history_daySpan << endl;

    // get sequences
    for(int i=0; i<flavorsinfo.vflavors.size(); i++)
    {
        vector<int> sequence(history_daySpan, 0); //history_daySpan ints with value 0;
        //search data
        for(int j=0; j<data_num; j++)
        {
            string line = data[j];
            int firstTab = line.find_first_of('\t');
            int secondTab = line.find_last_of('\t');
            int lastSpace = line.find_last_of(' ');
            string vflavor = line.substr(firstTab+sizeof("\t")-1, secondTab-firstTab-sizeof("\t")+1);
            time_t time =  str2time(line.substr(secondTab+sizeof("\t")-1, lastSpace-secondTab-1).append(" 00:00:00").c_str(), "%Y-%m-%d %H:%M:%S");
            int dayDiff = (time - history_begin_time_t)/oneDayLong; // used as sequence index
//            cout << vflavor << "\t" << time << "\t" << dayDiff << endl;
            if(vflavor == flavorsinfo.vflavors[i]) //this kind of flavor need to be predicted
            {
                sequence[dayDiff]++;   //add a record at daydiff index
            }
        }
        sequences.push_back(sequence);
    }

    for(int i=0; i<sequences.size(); i++)
    {
        cout << flavorsinfo.vflavors[i] << endl;
        int sum = 0;
        for(int j=0; j<sequences[i].size(); j++)
        {
            sum += sequences[i][j];
            cout << j << ": " << sequences[i][j] << ", ";
        }
        cout << "total_num: " << sum << endl;
        cout << endl;
    }
}

time_t str2time(string str, string format)
{
    struct tm *tmp_time = (struct tm*)malloc(sizeof(struct tm));
    strptime(str.c_str(), format.c_str(), tmp_time);
    time_t t = mktime(tmp_time);
    free(tmp_time);
    return t;
}
