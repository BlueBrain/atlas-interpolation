#!/usr/bin/env sh

# Strict execution:
# e: error in subcommand leads to exit of script
# u: unbound variables lead to errors
set -eu

VENDOR_DIR=src/atlinter/vendor

RIFE_URL=https://github.com/hzwer/arXiv2020-RIFE.git
RIFE_SHA=6ff174584737a9aa27cd9654443a8a79c76799c9
RIFE_SOURCE_DIR=model
RIFE_TARGET_DIR=rife

CAIN_URL=https://github.com/myungsub/CAIN.git
CAIN_SHA=fff8fc321c5a76904ed2a12c9500e055d4c77256
CAIN_SOURCE_DIR=model
CAIN_TARGET_DIR=cain


do_vendor()
{
  URL=$1
  SHA=$2
  SOURCE_DIR=$3
  TARGET_DIR=$4

  echo ">>> Vendoring '$SOURCE_DIR' -> '$TARGET_DIR' from $URL @ $SHA"

  TMPDIR=$(mktemp -d)
  git clone "$URL" "$TMPDIR"
  git -C "$TMPDIR" reset --hard "$SHA"
  cp -r "$TMPDIR/$SOURCE_DIR" "$VENDOR_DIR/$TARGET_DIR"
  rm -rf "$TMPDIR"
}

do_vendor $RIFE_URL $RIFE_SHA $RIFE_SOURCE_DIR $RIFE_TARGET_DIR
do_vendor $CAIN_URL $CAIN_SHA $CAIN_SOURCE_DIR $CAIN_TARGET_DIR

# In RIFE source files absolute imports need to be replaced by relative imports, e.g.
# "from model.IFNet_HD import *" -> "from .IFNet_HD import *"
for f in "$VENDOR_DIR"/"$RIFE_TARGET_DIR"/*.py
do
  sed -iE 's/^from model/from /g' "$f"
done

echo ">>> Vendoring finished successfully."
