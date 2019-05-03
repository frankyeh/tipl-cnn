#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <future>
#include <QMainWindow>
#include <QTimer>
#include <QGraphicsScene>
#include <QKeyEvent>
#include "tipl/tipl.hpp"
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    QTimer* timer = 0;
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    bool terminated;
    std::future<void> future;
    tipl::ml::trainer t;
    tipl::ml::network nn;
    tipl::ml::network_data<unsigned char> nn_data,nn_test;
    std::vector<unsigned int> train_seq;
    QGraphicsScene train_scene,data_scene;

    QImage train_image,data_image,data_image2;
protected:
    void keyPressEvent ( QKeyEvent * event );
private slots:
    void on_pushButton_clicked();

    void on_view_cnn_clicked();

    void on_actionOpen_data_triggered();

    void on_actionOpen_network_triggered();

    void on_actionOpen_testing_data_triggered();

    void on_actionSave_training_data_triggered();

    void on_actionSave_testing_data_triggered();

    void on_data_pos_valueChanged(int value);

    void on_actionCopy_X_Flip_triggered();

    void on_actionCopy_Y_Flip_triggered();

    void on_actionCopy_XY_Swap_triggered();

    void on_actionSave_network_triggered();

    void on_actionAdd_data_triggered();

    void on_actionIntensity_normalization_triggered();

    void on_reset_nn_clicked();

    void on_action10_fold_Data_Deparation_triggered();

    void on_classification_error_clicked();

    void on_actionPadding_rare_labels_triggered();

    void on_generate_list_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
