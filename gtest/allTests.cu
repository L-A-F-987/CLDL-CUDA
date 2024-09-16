#include "gtest/gtest.h"
using namespace std;


TEST(EmptyTest, test) {

}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
