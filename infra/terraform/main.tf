terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-southeast-2"
  default_tags {
    tags = { Project = "content-moderation" }   # tags every resource, handy for tracking cost
  }
}

# Looks up your account ID so the bucket name is globally unique
data "aws_caller_identity" "me" {}

# One bucket; models/, videos-raw/, videos-clean/ are just folders inside it
resource "aws_s3_bucket" "main" {
  bucket        = "sap-moderation-${data.aws_caller_identity.me.account_id}"
  force_destroy = true   # lets terraform destroy delete it even when it has files
}

# Blocks all public access, which is a security best practice for the report
resource "aws_s3_bucket_public_access_block" "main" {
  bucket                  = aws_s3_bucket.main.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Raw uploaded videos are deleted after 7 days, which is cost control
resource "aws_s3_bucket_lifecycle_configuration" "main" {
  bucket = aws_s3_bucket.main.id
  rule {
    id     = "expire-raw-videos"
    status = "Enabled"
    filter { prefix = "videos-raw/" }
    expiration { days = 7 }
  }
}

output "bucket_name" {
  value = aws_s3_bucket.main.bucket
}
