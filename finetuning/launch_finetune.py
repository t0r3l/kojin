"""Launch a Bedrock Custom Model fine-tuning job for Llama 3.1 8B.

Usage:
    python -m finetuning.launch_finetune \
        --s3-bucket YOUR_BUCKET \
        --role-arn arn:aws:iam::ACCOUNT:role/BedrockFineTuneRole \
        [--base-model meta.llama3-1-8b-instruct-v1:0] \
        [--region us-east-1] \
        [--epochs 3] \
        [--batch-size 4] \
        [--learning-rate 1e-5]

Prerequisites:
    1. Run `python -m finetuning.generate_dataset` first to produce train.jsonl & eval.jsonl
    2. Configure AWS credentials with permissions for:
       - s3:PutObject on the target bucket
       - bedrock:CreateModelCustomizationJob
       - iam:PassRole for the Bedrock service role
    3. The Bedrock service role must have s3:GetObject on the training data prefix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import boto3


def _upload_to_s3(local_path: str, bucket: str, key: str, region: str) -> str:
    """Upload a file to S3, return the s3:// URI."""
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def main():
    parser = argparse.ArgumentParser(description="Launch Bedrock fine-tuning job for Llama 3.1 8B")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket for training data")
    parser.add_argument("--s3-prefix", default="bedrock-finetune/kojin-nl2sql", help="S3 key prefix")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN for Bedrock customization")
    parser.add_argument("--base-model", default="amazon.nova-micro-v1:0:128k",
                        help="Base model ID (must be fine-tuning eligible in your region)")
    parser.add_argument("--region", default="us-east-1", help="AWS region for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size (Nova Micro accepts 1 only)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--job-name", default=None, help="Custom job name (auto-generated if omitted)")
    parser.add_argument("--data-dir", default="./finetuning/data", help="Local dir with train.jsonl/eval.jsonl")
    args = parser.parse_args()

    train_local = os.path.join(args.data_dir, "train.jsonl")
    eval_local = os.path.join(args.data_dir, "eval.jsonl")
    if not os.path.exists(train_local):
        print(f"ERROR: {train_local} not found. Run `python -m finetuning.generate_dataset` first.")
        sys.exit(1)

    job_name = args.job_name or f"kojin-nl2sql-{int(time.time())}"
    prefix = args.s3_prefix.strip("/")

    # Upload data
    print("Uploading training data to S3…")
    train_uri = _upload_to_s3(train_local, args.s3_bucket, f"{prefix}/train.jsonl", args.region)
    print(f"  train → {train_uri}")

    eval_uri = None
    if os.path.exists(eval_local):
        eval_uri = _upload_to_s3(eval_local, args.s3_bucket, f"{prefix}/eval.jsonl", args.region)
        print(f"  eval  → {eval_uri}")

    # Launch fine-tuning job
    bedrock = boto3.client("bedrock", region_name=args.region)

    training_config = {
        "trainingDataConfig": {"s3Uri": train_uri},
        "outputDataConfig": {"s3Uri": f"s3://{args.s3_bucket}/{prefix}/output/"},
    }
    if eval_uri:
        training_config["validationDataConfig"] = {"validators": [{"s3Uri": eval_uri}]}

    hyper_params = {
        "epochCount": str(args.epochs),
        "batchSize": str(args.batch_size),
        "learningRate": str(args.learning_rate),
    }

    print(f"\nLaunching fine-tuning job: {job_name}")
    print(f"  Base model : {args.base_model}")
    print(f"  Region     : {args.region}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.learning_rate}")

    response = bedrock.create_model_customization_job(
        jobName=job_name,
        customModelName=f"kojin-nl2sql-llama31-8b-{int(time.time())}",
        roleArn=args.role_arn,
        baseModelIdentifier=args.base_model,
        customizationType="FINE_TUNING",
        hyperParameters=hyper_params,
        **training_config,
    )

    job_arn = response.get("jobArn", "N/A")
    print(f"\n✓ Job créé : {job_arn}")
    print(f"  Suivre dans la console Bedrock → Custom models → Training jobs")
    print(f"  Ou via CLI : aws bedrock get-model-customization-job --job-identifier {job_arn} --region {args.region}")

    # Save job info for later use
    info_path = os.path.join(args.data_dir, "last_job.json")
    with open(info_path, "w") as f:
        json.dump({
            "job_name": job_name,
            "job_arn": job_arn,
            "base_model": args.base_model,
            "region": args.region,
            "train_uri": train_uri,
            "eval_uri": eval_uri,
        }, f, indent=2)
    print(f"  Info sauvegardée → {info_path}")


if __name__ == "__main__":
    main()
