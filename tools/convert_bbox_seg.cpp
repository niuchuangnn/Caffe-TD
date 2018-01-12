//
// Created by Niu Chuang on 17-11-19.
//

// This program converts a set of images and segmentation and bounding box annotations to a lmdb/leveldb by
// storing them as BBoxSegDatum proto buffers.
// Usage:
//   convert_bbox_seg [FLAGS] ROOTFOLDER/ SEGFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations of bounding box and segmentation, SEGFOLDER is the folder that holds segmentation mask images,
// and LISTFILE should be a list of files as well as their labels including bbox label file and
// segmentation mask label files if the mask label file exists or set to -1.
// The file should be in the format as
//   imgfolder1/img1.JPEG segfolder1/seg1.png annofolder1/anno1.xml
//   or
//   imgfolder1/img1.JPEG -1 annofolder1/anno1.xml

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

//#include <functional>
#include <tuple>
//#include <iostream>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::tuple;
using std::get;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
            "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
            "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
              "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(anno_type, "classification",
              "The type of annotation {classification, detection}.");
DEFINE_string(label_type, "xml",
              "The type of annotation file format.");
DEFINE_string(label_map_file, "",
              "A file with LabelMap protobuf message.");
DEFINE_bool(check_label, false,
            "When this option is on, check that there is no duplicated name/label.");
DEFINE_int32(min_dim, 0,
             "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
             "Maximum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
            "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
            "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
              "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Convert a set of images and annotations to the "
                                    "leveldb/lmdb format used as input for Caffe.\n"
                                    "Usage:\n"
                                    "    convert_annoset [FLAGS] ROOTFOLDER/ SEGFOLDER/ LISTFILE DB_NAME\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 5) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annoset");
        return 1;
    }

    const bool is_color = !FLAGS_gray;
    const bool check_size = FLAGS_check_size;
    const bool encoded = FLAGS_encoded;
    const string encode_type = FLAGS_encode_type;
    const string anno_type = FLAGS_anno_type;
//    AnnotatedDatum_AnnotationType type;
    const string label_type = FLAGS_label_type;
    const string label_map_file = FLAGS_label_map_file;
    const bool check_label = FLAGS_check_label;
    std::map<std::string, int> name_to_label;

    std::ifstream infile(argv[3]);
//    std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;
    std::vector<std::tuple<std::string, std::string, std::string> > lines;

    std::string imgname;
    int label;
    std::string bboxname;
    std::string segname;

    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
          << "Failed to convert name to label.";
    while (infile >> imgname >> segname >> bboxname) {
        lines.push_back(std::make_tuple(imgname, segname, bboxname));
    }

    if (FLAGS_shuffle) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        shuffle(lines.begin(), lines.end());
    }
    LOG(INFO) << "A total of " << lines.size() << " images.";

    if (encode_type.size() && !encoded)
        LOG(INFO) << "encode_type specified, assuming encoded=true.";

    int min_dim = std::max<int>(0, FLAGS_min_dim);
    int max_dim = std::max<int>(0, FLAGS_max_dim);
    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width = std::max<int>(0, FLAGS_resize_width);

    // Create new DB
    scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
    db->Open(argv[4], db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    std::string root_folder(argv[1]);
    std::string seg_folder(argv[2]);
//    AnnotatedDatum anno_datum;
//    Datum* datum = anno_datum.mutable_datum();
    BBoxSegDatum bbox_seg_datum;
    SegDatum* seg_datum = bbox_seg_datum.mutable_seg_datum();
    int count = 0;
    int data_size = 0;
    bool data_size_initialized = false;

    std::string enc_img;
    std::string enc_mask;
    for (int line_id = 0; line_id < lines.size(); ++line_id) {
        bool status = true;
        std::string enc = encode_type;


        // Guess the encoding type of image and ssegmentation mask from the file name
        if (!enc_img.size()) {
            string fn_img;
            fn_img = std::get<0>(lines[line_id]);
            size_t p = fn_img.rfind('.') + 1;
            if (p == fn_img.npos)
                LOG(WARNING) << "Failed to guess the encoding of '" << fn_img << "'";
            enc_img = fn_img.substr(p);
            std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        if (!enc_mask.size()) {
            string fn_mask;
            fn_mask = std::get<1>(lines[line_id]);
            size_t p1 = fn_mask.rfind('.') + 1;
            if (p1 == fn_mask.npos)
                LOG(WARNING) << "Failed to guess the encoding of '" << fn_mask << "'";
            enc_mask = fn_mask.substr(p1);
            std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }

        if (encoded && !enc.size()) {
            // Guess the encoding type from the file name
//            string fn = lines[line_id].first;
            string fn = std::get<0>(lines[line_id]);
            size_t p = fn.rfind('.');
            if ( p == fn.npos )
                LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
            enc = fn.substr(p);
            std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        imgname = root_folder + std::get<0>(lines[line_id]);
        if (std::get<1>(lines[line_id]) != "-1") {
            segname = seg_folder + std::get<1>(lines[line_id]);
        } else {
            segname = "-1";
        }
//        if (anno_type == "classification") {
//            label = boost::get<int>(lines[line_id].second);
//            status = ReadImageToDatum(filename, label, resize_height, resize_width,
//                                      min_dim, max_dim, is_color, enc, datum);
//        } else if (anno_type == "detection") {
//            labelname = root_folder + boost::get<std::string>(lines[line_id].second);
//            status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
//                                                   resize_width, min_dim, max_dim, is_color, enc, type, label_type,
//                                                   name_to_label, &anno_datum);
//            anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
//        }
//        if (status == false) {
//            LOG(WARNING) << "Failed to read " << lines[line_id].first;
//            continue;
//        }
        bboxname = root_folder + std::get<2>(lines[line_id]);
        status = ReadRichImageToBBoxSegDatum(imgname, segname, bboxname, resize_height, resize_width, min_dim, max_dim,
                                             is_color, enc_img, enc_mask, label_type, name_to_label, &bbox_seg_datum);
        if (!status) {
            LOG(WARNING) << "Failed to read " << std::get<0>(lines[line_id]);
            continue;
        }

        if (check_size) {
            if (!data_size_initialized) {
                data_size = seg_datum->channels() * seg_datum->height() * seg_datum->width();
                data_size_initialized = true;
            } else {
                const std::string& data = seg_datum->data();
                CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
                                                 << data.size();
            }
        }
        // sequential
        string key_str = caffe::format_int(line_id, 8) + "_" + std::get<0>(lines[line_id]);

        // Put in db
        string out;
        CHECK(bbox_seg_datum.SerializeToString(&out));
        txn->Put(key_str, out);

        if (++count % 1000 == 0) {
            // Commit db
            txn->Commit();
            txn.reset(db->NewTransaction());
            LOG(INFO) << "Processed " << count << " files.";
        }
    }
    // write the last batch
    if (count % 1000 != 0) {
        txn->Commit();
        LOG(INFO) << "Processed " << count << " files.";
    }
#else
    LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
    return 0;
}
