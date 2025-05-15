from models.modules.utils.net import PNet, RNet, ONet
from models import trainer
import os
if __name__ == '__main__':
    net = PNet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = trainer.Trainer(net, './param/p_net.pth', r"D:\CelebA\MTCN\64w\12")
    trainer.trainer(0.01)
    net = ONet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = trainer.Trainer(net, './param/o_net.pth', r"D:\CelebA\MTCN\64w\48")
    trainer.trainer(0.0003)
    net = RNet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = trainer.Trainer(net, './param/r_net.pth', r"D:\CelebA\MTCN\64w\\24")
    trainer.trainer(0.001)
