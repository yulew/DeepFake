from PIL import Image, ImageDraw
import random
import cv2
import numpy as np



def videos_split(table, train_or_test="train", fake_repeat_times=2, validation_split=0.2,
                 seed=543):  # (videos_dir, validation_split=0.2,seed=543): ###I will delete picking face features from .h5 features
    # 取出label==1，和label ==0的分两列，增多label==0的

    if train_or_test == "train":
        real_vfiles_list = list(table[table.label == "REAL"].index)
        fake_vfiles_list = list(table[table.label == "FAKE"].index)
        videos_files_list = real_vfiles_list + fake_repeat_times * fake_vfiles_list
        random.Random(seed).shuffle(videos_files_list)
        real_label_list = list(table[table.label == "REAL"].label)
        fake_label_list = list(table[table.label == "FAKE"].label)
        labels_list = list(real_label_list+fake_label_list)
        random.Random(seed).shuffle(labels_list)  # as they share the same random seed, so same random pattern
        # remove next four lines in next version
        real_facelocs_list = list(table[table.label == "REAL"].face_locations)
        fake_facelocs_list = list(table[table.label == "REAL"].fake_locations)
        face_locs_list = list(real_facelocs_list+fake_facelocs_list)
        random.Random(seed).shuffle(face_locs_list)
    else:#test, no labels, no need to fake repeatition
        videos_files_list = list(table.index)
        random.Random(seed).shuffle(videos_files_list)
        face_locs_list = list(table.face_locations)
        random.Random(seed).shuffle(videos_files_list)





    train_files_list = videos_files_list[:int((1 - validation_split) * len(videos_files_list))]
    val_files_list = videos_files_list[int((1 - validation_split) * len(videos_files_list)):]
    # remove next version
    train_locs_list = face_locs_list[:int((1 - validation_split) * len(face_locs_list))]
    val_locs_list = face_locs_list[int((1 - validation_split) * len(face_locs_list)):]

    if train_or_test == "train":
        train_labels_list = labels_list[:int((1 - validation_split) * len(labels_list))]
        val_labels_list = labels_list[int((1 - validation_split) * len(labels_list)):]
    else: # if test, the return train_labels_list or val_labels_list are empty
        train_labels_list = []
        val_labels_list = []
    # remove next two lines in next version

    return (train_files_list, train_labels_list, train_locs_list), (
        val_files_list, val_labels_list, val_locs_list)  # （，），（，）


def get_videos_generator(table, train_or_test="train", train_or_validation="train", validation_split=0.2,
                         videos_number_batch=32, seed=543):
    files_already = []  # record files already been loaded to prevent stop training/testing before loading all files
    i = 0
    if train_or_test == "train":
        train_list, val_list = videos_split(table, train_or_test=train_or_test, validation_split=validation_split, seed=seed)

        if train_or_validation == "train":
            videos_list = train_list  # videos_list includes train_files_list, train_label_list and face_locs_list
        else:
            videos_list = val_list
    else:  # train_or_test=="test"
        videos_list, _ = videos_split(table, validation_split=0, seed=seed)

    while i <= len(videos_list[0]):  # videos_list 包含三个list files_names, lables 和 locs，三个长度相同，所以videos_list[0]
        batch = (videos_list[0][i:i + videos_number_batch], videos_list[1][i:i + videos_number_batch],
                 videos_list[2][i:i + videos_number_batch])
        # record files already been loaded to prevent stop training/testing before loading all files in the list
        files_names_batch, _, _ = batch
        files_already += files_names_batch
        with open("files_names_arealdy_" + train_or_test + ".txt", "w") as handle:
            handle.write(str(files_already))
        i += videos_number_batch
        print(len(videos_list[0]), 'and', i)
        yield batch  # file_names, labels, face_locs


def get_videos_faces_generator(table, videos_dir, train_or_test="train", train_or_validation="train",
                               validation_split=0.2, videos_number_batch=32,
                               seed=543):  # 可以加一个argument每个视频挑几个训练，否则有太多重复

    gener = get_videos_generator(table=table, train_or_test=train_or_test, train_or_validation=train_or_validation,
                                 validation_split=validation_split, videos_number_batch=videos_number_batch, seed=seed)
    # i = 0

    if videos_dir[-1] != "/":
        video_dir = videos_dir + "/"
    while True:  # 当gener stop iteration
        batch_names, batch_labels, batch_face_locs = next(gener)
        batch_faces = []
        labels = []
        for (file, label, face_locs) in zip(batch_names, batch_labels, batch_face_locs):
            video_dir = videos_dir + file
            video_frames = get_frames(video_dir)
            # 暂时从表里抽取face localaton/face landmark. 正式版本要现运行, 以节省资源

            for (frame, face_loc) in zip(video_frames, face_locs):
                frame_face = get_face_given_location(frame, face_loc)
                frame_resized_face = face_resize(frame_face, face_len1=256, face_len2=256, n_channels=3)
                batch_faces.append(frame_resized_face)
                ####
                labels.append(label)
        random.Random(seed).shuffle(batch_faces)
        random.Random(seed).shuffle(labels)
        # i + = videos_number_batch
        yield np.array(batch_faces), labels

# 现在终于可以训练了， model.batch_fit()

