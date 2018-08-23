from resizeImage import fixData, loadAndGetData, getHorVert

#Størrelsen på bildet
img_horizontal = 356 ; img_vertical = 150

#Filplassering for treningsbilder
trainPath = "png/DTCtrain"
trainSavePath = 'train.npy'

#Filplassering for testbilder
testPath = "png/DTCtest"
testSavePath = 'test.npy'

#Shuffler bildene og lagrer de som en numpy array
fixData(img_horizontal, img_vertical, trainPath, 'train.npy')
train_y, train_x = loadAndGetData(trainSavePath)
fixData(img_horizontal, img_vertical, testPath, 'test.npy')
test_y, test_x = loadAndGetData(testSavePath)




