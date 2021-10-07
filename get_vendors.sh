#!/usr/bin/env sh

# Strict execution:
# e: error in subcommand leads to exit of script
# u: unbound variables lead to errors
set -eu

VENDOR_DIR=src/atlinter/vendor

RIFE_URL=https://github.com/hzwer/arXiv2020-RIFE.git
RIFE_SHA=6ff174584737a9aa27cd9654443a8a79c76799c9
RIFE_SOURCE_DIR='model/*'
RIFE_TARGET_DIR=rife

CAIN_URL=https://github.com/myungsub/CAIN.git
CAIN_SHA=fff8fc321c5a76904ed2a12c9500e055d4c77256
CAIN_SOURCE_DIR='model/*'
CAIN_TARGET_DIR=cain

MASKFLOWNET_URL=https://github.com/microsoft/MaskFlownet.git
MASKFLOWNET_SHA=5cba12772e2201f0d1c1e27161d224e585334571
MASKFLOWNET_SOURCE_DIR='predict_new_data.py:path.py:logger.py:network'
MASKFLOWNET_TARGET_DIR=MaskFlowNet

RAFT_URL=https://github.com/princeton-vl/RAFT.git
RAFT_SHA=224320502d66c356d88e6c712f38129e60661e80
RAFT_SOURCE_DIR="core/*"
RAFT_TARGET_DIR=RAFT

do_vendor()
{
  URL=$1
  SHA=$2
  SOURCE_DIR=$3
  TARGET_DIR=$4

  files=$(echo $SOURCE_DIR | tr ":" "\n")

  echo ">>> Vendoring '$SOURCE_DIR' -> '$TARGET_DIR' from $URL @ $SHA"

  TMPDIR=$(mktemp -d)
  git clone "$URL" "$TMPDIR"
  ls $TMPDIR
  git -C "$TMPDIR" reset --hard "$SHA"

  mkdir "$VENDOR_DIR/$TARGET_DIR"

  for file in $files
  do
    echo $TMPDIR/$file
    touch $VENDOR_DIR/$TARGET_DIR/__init__.py
    echo "mv $TMPDIR/$file '$VENDOR_DIR/$TARGET_DIR'"
    mv $TMPDIR/$file "$VENDOR_DIR/$TARGET_DIR"
  done

  rm -rf "$TMPDIR"
}

my_sed()
{
  # Run an in-place sed command on a file in a way that works both on
  # Linux and on macOS

  COMMAND=$1
  FILE=$2

  sed -E -e "$COMMAND" "$FILE" > "${FILE}.my_sed"
  mv "${FILE}.my_sed" "$FILE"
}

# Pair interpolation
do_vendor $RIFE_URL $RIFE_SHA $RIFE_SOURCE_DIR $RIFE_TARGET_DIR
do_vendor $CAIN_URL $CAIN_SHA $CAIN_SOURCE_DIR $CAIN_TARGET_DIR

# Optical Flow
do_vendor $RAFT_URL $RAFT_SHA $RAFT_SOURCE_DIR $RAFT_TARGET_DIR
do_vendor $MASKFLOWNET_URL $MASKFLOWNET_SHA $MASKFLOWNET_SOURCE_DIR $MASKFLOWNET_TARGET_DIR

# In RIFE source files absolute imports need to be replaced by relative imports, e.g.
# "from model.IFNet_HD import *" -> "from .IFNet_HD import *"
for f in "$VENDOR_DIR"/"$RIFE_TARGET_DIR"/*.py
do
  my_sed 's/^from model/from /g' "$f"
done

# RAFT
for f in "$VENDOR_DIR"/"$RAFT_TARGET_DIR"/*.py
do
  my_sed 's/^from update/from .update/g' "$f"
  my_sed 's/^from extractor/from .extractor/g' "$f"
  my_sed 's/^from corr/from .corr/g' "$f"
  my_sed 's/^from utils/from .utils/g' "$f"
done

# MaskFlowNet
for f in "$VENDOR_DIR"/"$MASKFLOWNET_TARGET_DIR"/*.py
do
  my_sed 's/^import path/from . import path/g' "$f"
  my_sed 's/^import logger/from . import logger/g' "$f"
  my_sed 's/^from network/from .network/g' "$f"
  my_sed 's/^import network.config/from .network import config/g' "$f"
  my_sed 's/network.config/config/g' "$f"
done
rm -r "$VENDOR_DIR"/"$MASKFLOWNET_TARGET_DIR"/network/config/*.yaml

touch $VENDOR_DIR/__init__.py
echo ">>> Vendoring finished successfully."
