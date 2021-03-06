import util
import numpy as np


def main():

    svm_mats=[ [[85265, 11],[36,131]], [[85268,11], [56,255]], [[85306,7],[81,344]], [[85292,12],[95,486]], [[85290,12],[134,597]],\
        [[85288,13],[166,714]], [[85300,11],[186,831]], [[85271,24],[174,1007]], [[85329,23],[206,1065]], [[85279,31],[233,1228]]]
    adb_mats=[ [[85293, 13], [36, 101]], [[85290, 18], [56, 226]], [[85287,33], [76,342]], [[85310,40], [106,429]], \
        [[85321,36], [123,553]], [[85299,49], [118,715]], [[85294, 44], [145,845]], [[85264,50], [193,969]], \
            [[85326,41], [208,1048]], [[85259,63], [214,1235]] ]
    fcnn_mat=[[[85273,11],[55, 104]], [[85269, 23],[63,235]], [[85279,34],[68,357]], [[85300,24],[108,453]], [[85300,29],[123,581]], \
        [[85289,26],[137, 729]], [[85292, 40],[136,860]], [[85253, 36],[223,964]], [[85353,21],[238,1011]],[[85267,42],[209,1253]]]
        
    model_mat_dict={'svm':svm_mats, 'adaboost':adb_mats, 'fcnn':fcnn_mat}
    upsample_range=np.arange(1,11)
    util.plot_recall_upsample_curve(model_mat_dict,upsample_range)

if __name__ == '__main__':
    main()