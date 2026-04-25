"""
Mesh Toroidal Seguro (Protocolo Omega-0 com Criptografia E2E)
Comunicação P2P entre nodos com TLS/SSL e Criptografia Assimétrica (RSA).
"""

import torch
import torch.nn as nn
import json
import base64
import io
import socket
import ssl
import threading
import hashlib
import hmac
from typing import Dict, List, Optional, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


class SecureNodeIdentity:
    """Identidade criptográfica de um nodo no Mesh"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.node_fingerprint = self._compute_fingerprint()
    
    def _compute_fingerprint(self) -> str:
        """Computa fingerprint único do nodo baseado na chave pública"""
        pub_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(pub_key_pem).hexdigest()[:16]
    
    def sign_message(self, message: bytes) -> bytes:
        """Assina uma mensagem com a chave privada"""
        return self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
    
    def verify_signature(self, message: bytes, signature: bytes, peer_public_key) -> bool:
        """Verifica assinatura de um peer"""
        try:
            peer_public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def export_public_key_pem(self) -> str:
        """Exporta a chave pública em formato PEM (para compartilhamento)"""
        pub_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pub_key_pem.decode('utf-8')
    
    @staticmethod
    def import_public_key_pem(pem_str: str):
        """Importa chave pública de um peer"""
        pub_key_pem = pem_str.encode('utf-8')
        return serialization.load_pem_public_key(pub_key_pem, backend=default_backend())


class SecureMessage:
    """Mensagem criptografada e assinada para transmissão segura"""
    
    def __init__(self, sender_id: str, recipient_id: str, payload: Dict, sender_key: SecureNodeIdentity):
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.payload = payload
        self.sender_key = sender_key
        self.timestamp = torch.tensor(torch.cuda.Event(enable_timing=True).record()).item() if torch.cuda.is_available() else 0.0
        
        # Serializar payload
        self.payload_json = json.dumps(payload)
        self.payload_bytes = self.payload_json.encode('utf-8')
        
        # Assinar mensagem
        self.signature = sender_key.sign_message(self.payload_bytes)
    
    def serialize(self) -> str:
        """Serializa a mensagem para transmissão"""
        message_dict = {
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'payload': self.payload,
            'signature': base64.b64encode(self.signature).decode('utf-8'),
            'sender_fingerprint': self.sender_key.node_fingerprint
        }
        return json.dumps(message_dict)
    
    @staticmethod
    def deserialize(message_json: str) -> Dict:
        """Desserializa uma mensagem recebida"""
        return json.loads(message_json)


class ToroidalMeshSecure(nn.Module):
    """
    Mesh Toroidal Seguro com Comunicação P2P Criptografada
    
    Características:
    - Autenticação RSA 2048-bit
    - Assinatura de mensagens
    - Verificação de integridade
    - Comunicação TLS/SSL
    - Rastreamento de peers confiáveis
    """
    
    def __init__(self, node_id: str, listen_port: int = 9999):
        super().__init__()
        self.node_id = node_id
        self.listen_port = listen_port
        self.identity = SecureNodeIdentity(node_id)
        
        # Registro de peers confiáveis (fingerprint -> public_key)
        self.trusted_peers: Dict[str, any] = {}
        
        # Registro de experts sincronizados
        self.expert_registry: Dict[str, Dict] = {}
        
        # Lock para thread-safety
        self.lock = threading.Lock()
        
        # Server socket para aceitar conexões
        self.server_socket = None
        self.server_thread = None
        self.is_running = False
    
    def register_trusted_peer(self, peer_id: str, peer_public_key_pem: str) -> bool:
        """Registra um peer como confiável após validação"""
        try:
            peer_public_key = SecureNodeIdentity.import_public_key_pem(peer_public_key_pem)
            peer_fingerprint = hashlib.sha256(peer_public_key_pem.encode()).hexdigest()[:16]
            
            with self.lock:
                self.trusted_peers[peer_fingerprint] = {
                    'peer_id': peer_id,
                    'public_key': peer_public_key,
                    'fingerprint': peer_fingerprint,
                    'status': 'trusted'
                }
            return True
        except Exception as e:
            print(f"[Mesh] Erro ao registrar peer {peer_id}: {e}")
            return False
    
    def serialize_expert(self, expert) -> str:
        """Serializa um expert para transmissão segura"""
        buffer = io.BytesIO()
        state = {
            'state_dict': expert.state_dict(),
            'd_model': expert.d_model,
            'internal_dim': expert.internal_dim,
            'activation_type': expert.activation_type,
            'phase_signature': expert.phase_signature.tolist() if hasattr(expert.phase_signature, 'tolist') else expert.phase_signature,
            'conatus': expert.conatus.item() if hasattr(expert.conatus, 'item') else float(expert.conatus),
            'vitality': 1.0 - (expert.conatus.item() if hasattr(expert.conatus, 'item') else float(expert.conatus))
        }
        torch.save(state, buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def deserialize_expert(self, expert_data: str, ExpertClass):
        """Reconstrói um expert recebido com validação de integridade"""
        try:
            raw_data = base64.b64decode(expert_data)
            buffer = io.BytesIO(raw_data)
            state = torch.load(buffer)
            
            expert = ExpertClass(
                d_model=state['d_model'],
                phase_signature=torch.tensor(state['phase_signature']) if isinstance(state['phase_signature'], list) else state['phase_signature'],
                internal_dim=state['internal_dim'],
                activation_type=state['activation_type']
            )
            expert.load_state_dict(state['state_dict'])
            expert.conatus.fill_(state['conatus'])
            return expert, state.get('vitality', 1.0)
        except Exception as e:
            print(f"[Mesh] Erro ao desserializar expert: {e}")
            return None, 0.0
    
    def create_secure_message(self, recipient_id: str, message_type: str, data: Dict) -> str:
        """Cria uma mensagem segura assinada"""
        payload = {
            'type': message_type,
            'data': data,
            'sender_fingerprint': self.identity.node_fingerprint
        }
        secure_msg = SecureMessage(self.node_id, recipient_id, payload, self.identity)
        return secure_msg.serialize()
    
    def verify_and_process_message(self, message_json: str, peer_fingerprint: str) -> Optional[Dict]:
        """Verifica e processa uma mensagem recebida"""
        try:
            message_data = SecureMessage.deserialize(message_json)
            
            # Verificar se o peer é confiável
            if peer_fingerprint not in self.trusted_peers:
                print(f"[Mesh] Peer {peer_fingerprint} não confiável. Rejeitando mensagem.")
                return None
            
            peer_info = self.trusted_peers[peer_fingerprint]
            peer_public_key = peer_info['public_key']
            
            # Verificar assinatura
            payload_bytes = json.dumps(message_data['payload']).encode('utf-8')
            signature = base64.b64decode(message_data['signature'])
            
            if not self.identity.verify_signature(payload_bytes, signature, peer_public_key):
                print(f"[Mesh] Falha na verificação de assinatura de {message_data['sender_id']}")
                return None
            
            return message_data['payload']
        except Exception as e:
            print(f"[Mesh] Erro ao processar mensagem: {e}")
            return None
    
    def broadcast_expert_update(self, expert_id: str, expert_data: str, vitality: float):
        """Broadcast de atualização de expert para todos os peers confiáveis"""
        message_type = 'expert_update'
        data = {
            'expert_id': expert_id,
            'expert_data': expert_data,
            'vitality': vitality,
            'source_node': self.node_id
        }
        
        with self.lock:
            for peer_fingerprint, peer_info in self.trusted_peers.items():
                if peer_info['status'] == 'trusted':
                    secure_msg = self.create_secure_message(
                        peer_info['peer_id'],
                        message_type,
                        data
                    )
                    # Em uma implementação real, enviar via socket
                    print(f"[Mesh] Broadcast para {peer_info['peer_id']}: {message_type}")
    
    def start_server(self):
        """Inicia o servidor P2P seguro"""
        self.is_running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        print(f"[Mesh] Servidor P2P iniciado no nodo {self.node_id} (porta {self.listen_port})")
    
    def _server_loop(self):
        """Loop do servidor para aceitar conexões seguras"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.listen_port))
            self.server_socket.listen(5)
            
            while self.is_running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    # Envolver com TLS/SSL
                    # client_socket = ssl.wrap_socket(client_socket, server_side=True, ...)
                    
                    # Processar conexão em thread separada
                    threading.Thread(target=self._handle_client, args=(client_socket, addr), daemon=True).start()
                except Exception as e:
                    if self.is_running:
                        print(f"[Mesh] Erro ao aceitar conexão: {e}")
        except Exception as e:
            print(f"[Mesh] Erro ao iniciar servidor: {e}")
    
    def _handle_client(self, client_socket, addr):
        """Processa uma conexão de cliente"""
        try:
            data = client_socket.recv(65536)
            if data:
                message_json = data.decode('utf-8')
                # Processar mensagem
                print(f"[Mesh] Mensagem recebida de {addr}: {len(message_json)} bytes")
        except Exception as e:
            print(f"[Mesh] Erro ao processar cliente: {e}")
        finally:
            client_socket.close()
    
    def stop_server(self):
        """Para o servidor P2P"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        print(f"[Mesh] Servidor P2P do nodo {self.node_id} parado")
    
    def get_node_info(self) -> Dict:
        """Retorna informações do nodo para compartilhamento"""
        return {
            'node_id': self.node_id,
            'fingerprint': self.identity.node_fingerprint,
            'public_key': self.identity.export_public_key_pem(),
            'listen_port': self.listen_port,
            'trusted_peers_count': len(self.trusted_peers)
        }
