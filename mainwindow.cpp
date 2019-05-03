#include <QFileDialog>
#include <QGraphicsTextItem>
#include <QMessageBox>
#include <QProgressDialog>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "gzip_interface.hpp"
extern std::shared_ptr<QProgressDialog> progressDialog;
extern std::vector<unsigned char> train_labels, test_labels;
extern std::vector<std::vector<float>> train_images, test_images;


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->graphicsView->setScene(&train_scene);
    ui->data_view->setScene(&data_scene);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::keyPressEvent ( QKeyEvent * event )
{
    if(!nn_data.empty() && ui->view_tab->currentIndex() == 1) // data page
    {
        switch(event->key())
        {
            case Qt::Key_Left:
            case Qt::Key_Up:
                ui->data_pos->setValue(ui->data_pos->value()-1);
                event->accept();
                break;
            case Qt::Key_Right:
            case Qt::Key_Down:
                ui->data_pos->setValue(ui->data_pos->value()+1);
                event->accept();
            break;
        }
        if(event->key() >= Qt::Key_0 && event->key() <= Qt::Key_9)
        {
            int label = event->key()-Qt::Key_0;
            nn_data.data_label[ui->data_pos->value()] = label;
            on_data_pos_valueChanged(0);
            event->accept();
        }
    }

    if(event->isAccepted())
        return;
    QWidget::keyPressEvent(event);

}

void MainWindow::on_reset_nn_clicked()
{
    if(future.valid())
    {
        terminated = true;
        future.wait();
    }
    nn.reset();
    if(ui->nn->text().isEmpty())
    {
        //64,80,3|conv,tanh,3|62,78,26|avg_pooling,tanh,2|31,39,26|full,tanh|1,1,120|full,tanh|1,1,56
        if(!(nn << "32,32,1|conv,relu,5|28,28,4|max_pooling,identity,2|14,14,4|full,relu|1,1,10"))
        {
            QMessageBox::information(this,"Error",QString("Invalid network text: %1").arg(nn.error_msg.c_str()));
            nn.reset();
            return;
        }
    }
    else
    {
        if(!(nn << ui->nn->text().toStdString()))
        {
            QMessageBox::information(this,"Error",QString("Invalid network text: %1").arg(nn.error_msg.c_str()));
            nn.reset();
            return;
        }
    }
    nn.init_weights();
}

void MainWindow::on_pushButton_clicked()
{
    using namespace tipl::ml;
    if(future.valid())
    {
        terminated = true;
        future.wait();
    }
    if(nn.empty())
        on_reset_nn_clicked();

    if(nn.empty())
        return;
    if(nn_data.input.size() != nn.get_input_size())
    {
        QMessageBox::information(this,"Error","Data does not fit the network dimension");
        return;
    }



    std::cout << "network cost:" << nn.computation_cost() << std::endl;
    terminated = false;
    future = std::async(std::launch::async, [&]
    {
        t.reset();
        t.learning_rate = ui->learning_rate->value()*0.001f;
        //t.w_decay = ui->w_decay->value();
        //t.b_decay = ui->b_decay->value();
        t.momentum = ui->momentum->value();
        t.batch_size = ui->batch_size->value();
        t.epoch = ui->epoch->value();
        int round = 0;
        t.error_table.resize(nn.get_output_size()*nn.get_output_size());
        network_data_proxy<unsigned char> proxy(nn_data);
        t.train(nn,proxy,terminated, [&](){
            //if(ui->rotate_sample->isChecked())
            //    nn_data.rotate_permute();
            if(!nn_test.empty())
            {
                tipl::ml::network_data_proxy<unsigned char> p(nn_test);
                nn.set_test_mode(true);
                std::vector<unsigned char> result(p.size());
                nn.predict(p,result);
                std::cout << round << " testing error:" << (float)p.calculate_miss(result)*100.0f/(float)result.size() << "%" << std::endl;
                nn.set_test_mode(false);
            }
            std::cout << round << " training error:" << t.get_training_error() << "%" << std::endl;
            std::cout << round << " decay = " << t.rate_decay << std::endl;
            ++round;
        });
    });


    if(timer)
        delete timer;
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(on_view_cnn_clicked()));
    timer->setInterval(2000);
    timer->start();
}

void MainWindow::on_view_cnn_clicked()
{
    if(nn_data.empty())
        return;
    ui->label->setText(QString("training error=%1%").arg(t.get_training_error()));
    tipl::color_image I2;
    static unsigned int i = 0;
    {
        if(i >= nn_data.data.size())
            i = 0;
        tipl::ml::to_image(nn,I2,nn_data.data[i],nn_data.data_label[i],20,(ui->graphicsView->width()-20)/2);
    }
    ++i;
    QImage qimage((unsigned char*)&*I2.begin(),I2.width(),I2.height(),QImage::Format_RGB32);
    train_image = qimage.scaled(I2.width()*2,I2.height()*2);
    train_scene.setSceneRect(0, 0, train_image.width(),train_image.height());
    train_scene.clear();
    train_scene.setItemIndexMethod(QGraphicsScene::NoIndex);
    train_scene.addRect(0, 0, train_image.width(),train_image.height(),QPen(),train_image);
}


void MainWindow::on_actionOpen_data_triggered()
{
    QString filename;
    filename = QFileDialog::getOpenFileName(
                this,
                "Open data","network_data.bin.gz",
                "Data files (*.bin.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    begin_prog("reading");
    nn_data.load_from_file<gz_istream>(filename.toStdString().c_str());
    ui->data_pos->setMaximum(nn_data.size()-1);
    filename.replace("train","test");
}

void MainWindow::on_actionOpen_network_triggered()
{
    QString filename;
    filename = QFileDialog::getOpenFileName(
                this,
                "Open data","network.net.gz",
                "Network files (*.net.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    begin_prog("reading");
    nn.load_from_file<gz_istream>(filename.toStdString().c_str());
    ui->nn->setText(nn.get_layer_text().c_str());
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(on_view_cnn_clicked()));
    timer->setInterval(2000);
    timer->start();
}

void MainWindow::on_actionSave_network_triggered()
{
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save network","network.net.gz",
                "Network files (*.net.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    nn.save_to_file<gz_ostream>(filename.toStdString().c_str());
}
void MainWindow::on_actionOpen_testing_data_triggered()
{
    QString filename;
    filename = QFileDialog::getOpenFileName(
                this,
                "Open data","network_data.bin.gz",
                "Data files (*.bin.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    begin_prog("reading");
    nn_test.load_from_file<gz_istream>(filename.toStdString().c_str());
}

void MainWindow::on_actionSave_training_data_triggered()
{
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save data","network_data.bin.gz",
                "Data files (*.bin.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    nn_data.save_to_file<gz_ostream>(filename.toStdString().c_str());

}

void MainWindow::on_actionSave_testing_data_triggered()
{
    QString filename;
    filename = QFileDialog::getSaveFileName(
                this,
                "Save data","network_data.bin.gz",
                "Data files (*.bin.gz);;All files (*)");
    if(filename.isEmpty())
        return;
    nn_test.save_to_file<gz_ostream>(filename.toStdString().c_str());
}


void MainWindow::on_data_pos_valueChanged(int value)
{
    if(nn_data.empty())
        return;
    auto& data = nn_data.data[ui->data_pos->value()];
    tipl::grayscale_image I(tipl::geometry<2>(nn_data.input[0],nn_data.input[1]*nn_data.input[2]));
    for(int i = 0;i < data.size();++i)
        I[i] = std::floor((data[i]+1.0f)*127.0f);
    tipl::color_image cI = I;
    QImage qimage((unsigned char*)&*cI.begin(),cI.width(),cI.height(),QImage::Format_RGB32);



    data_image = qimage.scaledToHeight(std::max<int>(qimage.height(),data_scene.views()[0]->height()-50));
    data_scene.setSceneRect(0, 0, data_image.width()*2,data_image.height()+20);
    data_scene.clear();
    data_scene.setItemIndexMethod(QGraphicsScene::NoIndex);
    data_scene.addRect(0, 0, data_image.width(),data_image.height(),QPen(),data_image);

    if(nn_data.input[2] == 3) // RGB
    {
        tipl::color_image I2(tipl::geometry<2>(nn_data.input[0],nn_data.input[1]));
        int shift1 = I2.size();
        int shift2 = shift1 + shift1;
        for(int i = 0;i < I2.size();++i)
            I2[i] = tipl::rgb(I[i],I[i+shift1],I[i+shift2]);
        QImage qimage2((unsigned char*)&*I2.begin(),I2.width(),I2.height(),QImage::Format_RGB32);
        data_image2 = qimage2.scaledToWidth(data_image.width());
        data_scene.addRect(data_image.width(), 0, data_image2.width(),data_image2.height(),QPen(),data_image2);
    }
    QString info;
    info = QString("%1/%2 Label=%3").arg(ui->data_pos->value()).
                                     arg(nn_data.size()).
                                     arg((int)nn_data.data_label[ui->data_pos->value()]);
    if(!nn.empty() && nn_data.input.size() == nn.get_input_size())
    {
        unsigned char label = 0;
        nn.predict(nn_data.data[ui->data_pos->value()],label);
        info += QString(" NN:%1").arg(label);
    }

    QGraphicsTextItem* text = data_scene.addText(info);
    text->moveBy(0,data_image.height());
}

void MainWindow::on_actionCopy_X_Flip_triggered()
{
    size_t size = nn_data.size();
    for(int i = 0;i < size;++i)
    {
        nn_data.data_label.push_back(nn_data.data_label[i]);
        nn_data.data.push_back(nn_data.data[i]);
        auto I= tipl::make_image(&nn_data.data.back()[0],
                tipl::geometry<3>(nn_data.input[0],nn_data.input[1],nn_data.input[2]));
        tipl::flip_x(I);
    }
    ui->data_pos->setMaximum(nn_data.size()-1);
}

void MainWindow::on_actionCopy_Y_Flip_triggered()
{
    size_t size = nn_data.size();
    for(int i = 0;i < size;++i)
    {
        nn_data.data_label.push_back(nn_data.data_label[i]);
        nn_data.data.push_back(nn_data.data[i]);
        auto I = tipl::make_image(&nn_data.data.back()[0],
                tipl::geometry<3>(nn_data.input[0],nn_data.input[1],nn_data.input[2]));
        tipl::flip_y(I);

    }
    ui->data_pos->setMaximum(nn_data.size()-1);
}

void MainWindow::on_actionCopy_XY_Swap_triggered()
{
    size_t size = nn_data.size();
    for(int i = 0;i < size;++i)
    {
        nn_data.data_label.push_back(nn_data.data_label[i]);
        nn_data.data.push_back(nn_data.data[i]);
        auto I = tipl::make_image(&nn_data.data.back()[0],
                tipl::geometry<3>(nn_data.input[0],nn_data.input[1],nn_data.input[2]));
        tipl::swap_xy(I);
    }
    ui->data_pos->setMaximum(nn_data.size()-1);
}


void MainWindow::on_actionAdd_data_triggered()
{
    QString filename;
    filename = QFileDialog::getOpenFileName(
                this,
                "Open data","data.bin",
                "Data files (*.bin);;All files (*)");
    if(filename.isEmpty())
        return;
    tipl::ml::network_data<unsigned char> new_data;
    begin_prog("reading");
    new_data.load_from_file<gz_istream>(filename.toStdString().c_str());
    if(new_data.input != nn_data.input)
    {
        QMessageBox::information(this,"error","Inconsistent input dimension",0);
        return;
    }
    for(int i = 0;i < new_data.data.size();++i)
    {
        nn_data.data.push_back(new_data.data[i]);
        nn_data.data_label.push_back(new_data.data_label[i]);
    }
    ui->data_pos->setMaximum(nn_data.data.size()-1);
}

void MainWindow::on_actionIntensity_normalization_triggered()
{
    for(int i = 0;i < nn_data.data.size();++i)
        tipl::normalize_abs(nn_data.data[i]);
}



void MainWindow::on_action10_fold_Data_Deparation_triggered()
{
    if(nn_data.empty())
        return;
    std::vector<int> order(nn_data.size());
    for(int i = 0;i < order.size();++i)
        order[i] = i;
    std::random_shuffle(order.begin(),order.end());
    tipl::ml::network_data<unsigned char> new_train_data,new_test_data;

    for(int i = 0;i < order.size();++i)
        if(i % 10 == 0)
        {
            new_test_data.data.push_back(nn_data.data[order[i]]);
            new_test_data.data_label.push_back(nn_data.data_label[order[i]]);
        }
    else
        {
            new_train_data.data.push_back(nn_data.data[order[i]]);
            new_train_data.data_label.push_back(nn_data.data_label[order[i]]);
        }
    nn_data.data = new_train_data.data;
    nn_data.data_label = new_train_data.data_label;
    nn_test.input = nn_data.input;
    nn_test.output = nn_data.output;
    nn_test.data = new_test_data.data;
    nn_test.data_label = new_test_data.data_label;
    ui->data_pos->setMaximum(nn_data.size()-1);
}

void MainWindow::on_classification_error_clicked()
{
    if(!t.error_table.empty())
    {
        std::ostringstream out;
        for(int i = 0,pos = 0;i < nn.get_output_size();++i)
        {
            out << "class" << i;
            for(int j = 0;j < nn.get_output_size();++j,++pos)
            {
                out << std::setw(5) << t.error_table[pos];
            }
            out << std::endl;
        }
        ui->error_table->setText(out.str().c_str());
    }
}

void MainWindow::on_actionPadding_rare_labels_triggered()
{
    if(nn_data.data.empty())
        return;
    unsigned int label_num = *std::max_element(nn_data.data_label.begin(),nn_data.data_label.end()) + 1;
    std::vector<unsigned int> label_count(label_num);
    for(int i = 0;i < nn_data.data_label.size();++i)
        ++label_count[nn_data.data_label[i]];
    unsigned int min_size = nn_data.data.size()/label_num/5;
    std::cout << "min size=" << min_size << std::endl;
    for(int i = 0;i < label_num;++i)
    {

        while(label_count[i] < min_size)
        {
            std::cout << "padding label" << i << " size = " << label_count[i] << std::endl;
            int size = nn_data.data_label.size();
            for(int j = 0;j < size;++j)
                if(nn_data.data_label[j] == i)
                {
                    nn_data.data.push_back(nn_data.data[j]);
                    nn_data.data_label.push_back(i);
                }
            label_count[i] *= 2;
        }
    }
    ui->data_pos->setMaximum(nn_data.size()-1);
}

void MainWindow::on_generate_list_clicked()
{
    if(nn_data.data.empty())
        return;
    std::vector<std::string> list;
    tipl::ml::iterate_cnn(nn_data.input,nn_data.output,list);
    std::cout << list.size() << std::endl;
    QString result;
    for(int i = 0;i < list.size() && i < 5000;++i)
    {
        result += list[i].c_str();
        result += "\r\n";
    }
    ui->network_list->setText(result);

}
