#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/group/pmc085/hlionar/cvs-domain-ssl"
OUTPUT_ROOT="/group/pmc085/hlionar/outputs/cvs-domain-ssl"
BACKUP_ROOT="/group/pmc085/hlionar/backups/cvs-domain-ssl"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/${STAMP}"
ARCHIVE="${BACKUP_ROOT}/cvs-domain-ssl-backup-${STAMP}.tar.gz"

mkdir -p "${BACKUP_DIR}"

echo "Project root: ${PROJECT_ROOT}"
echo "Output root:  ${OUTPUT_ROOT}"
echo "Backup dir:   ${BACKUP_DIR}"

cd "${PROJECT_ROOT}"

echo "Saving git status..."
git status -sb > "${BACKUP_DIR}/git_status.txt"
git log --oneline --decorate -30 > "${BACKUP_DIR}/git_log_last30.txt"

echo "Creating git bundle..."
git bundle create "${BACKUP_DIR}/cvs-domain-ssl.git.bundle" --all

echo "Copying repository metadata and scripts..."
mkdir -p "${BACKUP_DIR}/repo"
rsync -av \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  "${PROJECT_ROOT}/" "${BACKUP_DIR}/repo/"

echo "Copying experiment histories and configs..."
mkdir -p "${BACKUP_DIR}/outputs"
find "${OUTPUT_ROOT}" \
  \( -name "history.json" -o -name "config.json" -o -name "*.out" -o -name "*.err" \) \
  -print0 | rsync -av --files-from=- --from0 / "${BACKUP_DIR}/outputs/"

echo "Copying best checkpoints if available..."
find "${OUTPUT_ROOT}" \
  -name "best.pt" \
  -print0 | rsync -av --files-from=- --from0 / "${BACKUP_DIR}/outputs/" || true

echo "Creating archive..."
tar -czf "${ARCHIVE}" -C "${BACKUP_ROOT}" "${STAMP}"

echo
echo "Backup complete:"
echo "${ARCHIVE}"
echo
echo "IMPORTANT: this archive is still on Kaya."
echo "Before the shutdown, copy it to your local computer or IRDS."
