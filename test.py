import torch
import datetime
from dataset import *
from utils_pose import show_pose, modify_20_keypoints
from moviepy.editor import *


def test(model_dir, testset, descriptions, video_save_dir):
    dataset = TestDataset(test_samles_dir=testset)
    testloader = DataLoaderX(dataset=dataset, batch_size=1)
    print('loading model...')
    G = torch.load(model_dir).cuda()
    G.eval()

    for step, (music_feature, name) in enumerate(testloader):
        name = name[0]
        print('evaluating {}/{} '.format(step,len(dataset)),name)

        music_feature = music_feature.transpose(1,2)
        var_x = music_feature.float().cuda()
        predicted_pose, hx= G(var_x, None)

        predicted_pose = predicted_pose[0].detach().cpu().numpy()/3.5
        predicted_pose_mean = np.mean(predicted_pose,axis=0)
        predicted_pose-=predicted_pose_mean
        predicted_pose = modify_20_keypoints(predicted_pose)
        keypoints_mean = np.load('keypoints_mean.npy', allow_pickle=True)
        predicted_pose += keypoints_mean

        np.save(video_save_dir+name,predicted_pose)

        print('rendering video...')
        show_pose([predicted_pose], descriptions=descriptions, name=name, video_save_dir = video_save_dir)

        print('mix audio and video...')
        video = VideoFileClip(video_save_dir + name + '.avi')
        video = video.set_audio((AudioFileClip(testset+name)))
        video.write_videofile(video_save_dir + name + '.mp4')
        os.remove(video_save_dir + name + '.avi')
    print('test finished')


if __name__ == '__main__':
    model_dir = 'checkpoints/G_globalstep17000.pt'
    time_stamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    video_save_dir = 'test\\results\\'+'test_result_'+time_stamp+'/'
    os.mkdir(video_save_dir)
    test(model_dir=model_dir,
         testset='test\\testset\\',
         descriptions=['percep+adv'],
         video_save_dir=video_save_dir)
