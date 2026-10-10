variable "my_ip" {
  type = string
}

variable "instance_type" {
  default = "t3.large" # 2 vCPU, 8 GB RAM
}

# Use the account's default network, so no NAT gateway (which would cost money)
data "aws_vpc" "default" {
  default = true
}

# Latest official Ubuntu 24.04 image
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
}

resource "aws_key_pair" "main" {
  key_name   = "moderation-key"
  public_key = file(pathexpand("~/.ssh/moderation.pub"))
}

# Firewall: SSH only from your IP; web traffic from anywhere
resource "aws_security_group" "k3s" {
  name        = "moderation-k3s"
  description = "SSH from my IP, HTTP/HTTPS public"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# IAM role the server uses: it can only read/write OUR bucket (least privilege)
resource "aws_iam_role" "node" {
  name = "moderation-node-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "s3" {
  name = "s3-bucket-access"
  role = aws_iam_role.node.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.main.arn },
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], Resource = "${aws_s3_bucket.main.arn}/*" }
    ]
  })
}

resource "aws_iam_instance_profile" "node" {
  name = "moderation-node-profile"
  role = aws_iam_role.node.name
}

# The server itself; k3s installs automatically on first boot
resource "aws_instance" "k3s" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.main.key_name
  vpc_security_group_ids = [aws_security_group.k3s.id]
  iam_instance_profile   = aws_iam_instance_profile.node.name

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  user_data = <<-EOT
    #!/bin/bash
    curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
  EOT

  tags = { Name = "moderation-k3s" }
}

output "public_ip" {
  value = aws_instance.k3s.public_ip
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/moderation ubuntu@${aws_instance.k3s.public_ip}"
}
